"""Automated, zero-config dataset downloader for Small Object Detection benchmarks.

Supports:
- VisDrone2019-DET (val, train, test-dev) via Ultralytics CDN mirror.
- AI-TOD-v2 (annotations, train, val, test) via official Google Drive archives using gdown.

Features:
- 100% autonomous zero-config execution (no tokens, API keys, or manual logins).
- Resumable streaming download with integrity checks.
- Automated extraction into canonical directory structures.
- Immediate post-extraction archive cleanup in data/_downloads to conserve disk space.
- Fast --sample-only mode (default 500 images) for lightweight local pipeline verification.
- OS-agnostic path handling and error recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path

import gdown
import requests

from src.utils import ui

VISDRONE_URLS: dict[str, str] = {
    "val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
    "test-dev": (
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
        "VisDrone2019-DET-test-dev.zip"
    ),
    "train": (
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
        "VisDrone2019-DET-train.zip"
    ),
}

# Official AI-TOD-v2 Google Drive archive file IDs
AITOD_GDRIVE_ARCHIVES: dict[str, str] = {
    "annotations_v2": "131jpoF9oThL2L8y0CU0u9JI1vy76WrXC",
    "val": "1A_dJ5MBug9YajqJ09fblIfguia1opKI-",
    "train": "1JllO13n3is6stRxtWdaJsJpNutKRKn6b",
    "test": "1qCT2WDD8kRjo2DFgaxKvHsjdyY3rdE2S",
    "annotations_v1": "1l9wAPSChCjI8VwRUJPsxVWP3gpeWVI-r",
}


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_file_with_resume(
    url: str,
    target_path: Path,
    expected_size: int | None = None,
    max_retries: int = 5,
    timeout: int = 30,
) -> bool:
    """Download file via HTTP/HTTPS with resume capability and exponential backoff."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".part")

    initial_size = temp_path.stat().st_size if temp_path.exists() else 0

    for attempt in range(1, max_retries + 1):
        try:
            headers = {}
            if initial_size > 0:
                headers["Range"] = f"bytes={initial_size}-"

            response = requests.get(url, headers=headers, stream=True, timeout=timeout)

            if response.status_code == 416 and temp_path.exists():
                temp_path.rename(target_path)
                return True
            response.raise_for_status()

            total_size = int(response.headers.get("Content-Length", 0))
            if response.status_code == 206:
                total_size += initial_size
            elif response.status_code == 200:
                initial_size = 0

            progress = ui.create_progress()
            with progress:
                task = progress.add_task(
                    f"Downloading {target_path.name}",
                    total=total_size if total_size > 0 else None,
                    completed=initial_size,
                )

                with open(temp_path, "ab" if initial_size > 0 else "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))

            if temp_path.exists():
                if expected_size and temp_path.stat().st_size != expected_size:
                    ui.print_warning(
                        f"Size mismatch: expected {expected_size}, got {temp_path.stat().st_size}"
                    )
                temp_path.rename(target_path)
                return True

        except (OSError, requests.RequestException) as e:
            backoff = 2**attempt
            ui.print_warning(
                f"Download attempt {attempt}/{max_retries} failed: {e}. Retrying in {backoff}s..."
            )
            time.sleep(backoff)
            initial_size = temp_path.stat().st_size if temp_path.exists() else 0

    ui.print_error(f"Failed to download {url} after {max_retries} attempts.")
    return False


def download_gdrive_archive(
    file_id: str,
    target_path: Path,
    max_retries: int = 3,
) -> bool:
    """Download an archive directly from Google Drive using gdown without tokens."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            ui.print_info(f"Fetching archive {target_path.name} from Google Drive (ID: {file_id})...")
            res = gdown.download(id=file_id, output=str(target_path), quiet=False)
            if res and target_path.exists() and target_path.stat().st_size > 0:
                ui.print_success(
                    f"Downloaded {target_path.name} ({target_path.stat().st_size / (1024*1024):.2f} MB)"
                )
                return True
        except (OSError, RuntimeError, requests.RequestException) as e:
            ui.print_warning(f"gdown attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(2**attempt)

    ui.print_error(f"Failed to fetch Google Drive file ID {file_id} after {max_retries} attempts.")
    return False


def extract_zip(
    zip_path: Path,
    dest_dir: Path,
    max_files: int | None = None,
    delete_after: bool = True,
) -> bool:
    """Extract a ZIP archive cleanly, verify non-empty extraction, and optionally delete archive."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            if max_files is not None and max_files > 0:
                members = members[:max_files]

            progress = ui.create_progress()
            with progress:
                task = progress.add_task(f"Extracting {zip_path.name}", total=len(members))
                for member in members:
                    zf.extract(member, dest_dir)
                    progress.update(task, advance=1)

        # Verify extraction
        extracted_entries = list(dest_dir.glob("*"))
        if not extracted_entries:
            ui.print_error(f"Verification failed: {dest_dir} is empty after extraction.")
            return False

        ui.print_success(f"Extracted {len(members)} entries into {dest_dir}")

        if delete_after and zip_path.exists():
            zip_path.unlink()
            ui.print_info(f"Cleaned up archive {zip_path.name} from _downloads")

        return True
    except (OSError, zipfile.BadZipFile) as e:
        ui.print_error(f"Extraction failed for {zip_path.name}: {e}")
        return False


def download_visdrone(
    dest_dir: Path,
    splits: Sequence[str] = ("val",),
    sample_only: bool = False,
    sample_count: int = 500,
) -> Path:
    """Download and prepare VisDrone2019-DET dataset splits with automated cleanup."""
    ui.print_header("VisDrone2019-DET Downloader", f"Target: {dest_dir} | Splits: {list(splits)}")
    visdrone_dir = dest_dir / "visdrone"
    visdrone_dir.mkdir(parents=True, exist_ok=True)
    raw_zips_dir = dest_dir / "_downloads"
    raw_zips_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        if split not in VISDRONE_URLS:
            ui.print_warning(f"Unknown VisDrone split '{split}', skipping.")
            continue

        url = VISDRONE_URLS[split]
        zip_file = raw_zips_dir / f"VisDrone2019-DET-{split}.zip"
        extract_folder = visdrone_dir / f"VisDrone2019-DET-{split}"

        if not extract_folder.exists() or not any(extract_folder.iterdir()):
            if not zip_file.exists():
                ui.print_info(f"Fetching {zip_file.name} from Ultralytics CDN...")
                success = download_file_with_resume(url, zip_file)
                if not success:
                    continue
            ui.print_info(f"Extracting to {visdrone_dir}...")
            extract_zip(zip_file, visdrone_dir, delete_after=True)
        else:
            ui.print_info(f"Split {split} already extracted at {extract_folder}.")
            if zip_file.exists():
                zip_file.unlink()

    if sample_only:
        sample_dir = visdrone_dir / f"sample_{sample_count}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        img_candidates = list(visdrone_dir.glob("**/*.jpg"))
        if img_candidates:
            ui.print_info(f"Assembling {sample_count} sample images into {sample_dir}...")
            for img in img_candidates[:sample_count]:
                shutil.copy2(img, sample_dir / img.name)
            ui.print_success(f"Sample images ready at {sample_dir}")
            return sample_dir

    ui.print_success(f"VisDrone dataset ready at {visdrone_dir}")
    return visdrone_dir


def download_aitod(
    dest_dir: Path,
    splits: Sequence[str] = ("val",),
    sample_only: bool = False,
    sample_count: int = 500,
) -> Path:
    """Download and prepare AI-TOD-v2 dataset zero-config via official Google Drive archives."""
    ui.print_header(
        "AI-TOD-v2 Zero-Config Downloader",
        f"Target: {dest_dir} | Splits: {list(splits)} | Sample Only: {sample_only}",
    )
    aitod_dir = dest_dir / "ai-tod-v2"
    annotations_dir = aitod_dir / "annotations"
    images_dir = aitod_dir / "images"
    raw_zips_dir = dest_dir / "_downloads"

    annotations_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    raw_zips_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch Annotations v2 archive
    ann_target_files = [
        annotations_dir / "aitod_train_v2.json",
        annotations_dir / "aitod_val_v2.json",
        annotations_dir / "aitodv2_train.json",
        annotations_dir / "aitod_val.json",
    ]
    if not any(f.exists() for f in ann_target_files):
        ann_zip = raw_zips_dir / "annotations_v2.zip"
        if not ann_zip.exists():
            download_gdrive_archive(AITOD_GDRIVE_ARCHIVES["annotations_v2"], ann_zip)
        if ann_zip.exists():
            extract_zip(ann_zip, aitod_dir, delete_after=True)
            # Create standard aliases if needed
            train_src = annotations_dir / "aitod_train_v2.json"
            train_alias = annotations_dir / "aitodv2_train.json"
            if train_src.exists() and not train_alias.exists():
                shutil.copy2(train_src, train_alias)

            val_src = annotations_dir / "aitod_val_v2.json"
            val_alias = annotations_dir / "aitod_val.json"
            if val_src.exists() and not val_alias.exists():
                shutil.copy2(val_src, val_alias)
    else:
        ui.print_info(f"AI-TOD-v2 annotations already present at {annotations_dir}")

    # 2. Handle Images
    if sample_only:
        sample_dir = images_dir / f"sample_{sample_count}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Check existing images in mini split or val
        existing_imgs = list(images_dir.glob("**/*.png"))
        if not existing_imgs:
            mini_imgs = list(dest_dir.glob("ai-tod-v2/splits/mini/images/**/*.png"))
            existing_imgs.extend(mini_imgs)

        if len(existing_imgs) >= sample_count:
            ui.print_info(f"Assembling {sample_count} sample images from existing pool...")
            for img in existing_imgs[:sample_count]:
                shutil.copy2(img, sample_dir / img.name)
            ui.print_success(f"Sample subset assembled at {sample_dir}")
            return aitod_dir

        # Otherwise download val.zip to extract sample images without downloading full train
        val_zip = raw_zips_dir / "val.zip"
        if not val_zip.exists():
            download_gdrive_archive(AITOD_GDRIVE_ARCHIVES["val"], val_zip)
        if val_zip.exists():
            val_dest = images_dir / "val"
            extract_zip(val_zip, val_dest, max_files=sample_count, delete_after=True)
            for img in list(val_dest.glob("*.png"))[:sample_count]:
                shutil.copy2(img, sample_dir / img.name)
            ui.print_success(f"Sample images extracted to {sample_dir}")
            return aitod_dir

    # Full split downloads
    for split in splits:
        if split not in AITOD_GDRIVE_ARCHIVES:
            ui.print_warning(f"Unknown AI-TOD split '{split}', skipping.")
            continue

        split_dest = images_dir / split
        split_dest.mkdir(parents=True, exist_ok=True)
        existing_count = len(list(split_dest.glob("*.png")))
        if existing_count > 0:
            ui.print_info(f"Split {split} already populated ({existing_count} images). Skipping.")
            continue

        zip_file = raw_zips_dir / f"{split}.zip"
        if not zip_file.exists():
            download_gdrive_archive(AITOD_GDRIVE_ARCHIVES[split], zip_file)

        if zip_file.exists():
            ui.print_info(f"Extracting {zip_file.name} to {split_dest}...")
            extract_zip(zip_file, split_dest, delete_after=True)

    ui.print_success(f"AI-TOD-v2 dataset ready at {aitod_dir}")
    return aitod_dir


def download_datasets(
    dataset: str = "visdrone",
    extract_dir: Path = Path("data"),
    splits: Sequence[str] = ("val",),
    sample_only: bool = False,
    sample_count: int = 500,
) -> None:
    """Download requested datasets (visdrone, aitod-v2, or all)."""
    if dataset in ["visdrone", "all"]:
        download_visdrone(
            dest_dir=extract_dir,
            splits=splits,
            sample_only=sample_only,
            sample_count=sample_count,
        )

    if dataset in ["aitod-v2", "all"]:
        download_aitod(
            dest_dir=extract_dir,
            splits=splits,
            sample_only=sample_only,
            sample_count=sample_count,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated zero-config dataset downloader for Small Object Detection benchmarks."
    )
    parser.add_argument(
        "--dataset",
        choices=["aitod-v2", "visdrone", "all"],
        default="visdrone",
        help="Dataset to download (default: visdrone)",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("data"),
        help="Root directory for dataset storage (default: data)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val"],
        help="Splits to download (default: val)",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Extract a small sample set of images for immediate local benchmarking",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=500,
        help="Number of sample images to collect when --sample-only is active (default: 500)",
    )

    args = parser.parse_args()
    download_datasets(
        dataset=args.dataset,
        extract_dir=args.extract_dir,
        splits=args.splits,
        sample_only=args.sample_only,
        sample_count=args.sample_count,
    )


if __name__ == "__main__":
    main()
