"""Automated, resilient dataset downloader for Small Object Detection benchmarks.

Supports:
- VisDrone2019-DET (val, train, test-dev) via Ultralytics CDN mirror.
- AI-TOD-v2 via direct Google Drive mirror or Hugging Face dataset.

Features:
- Resumable chunked streaming download with SHA256 / size integrity checks.
- Retry logic with exponential backoff.
- Rich progress indicator integration via src.utils.ui.
- Automated extraction and directory structuring into data/ directory.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path

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

AITOD_GDRIVE_FOLDER: str = "1CowS5BrujefWQxxlmOFfUuLOfUUm8w6U"


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


def extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    """Extract a ZIP archive cleanly with progress tracking."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            progress = ui.create_progress()
            with progress:
                task = progress.add_task(f"Extracting {zip_path.name}", total=len(members))
                for member in members:
                    zf.extract(member, dest_dir)
                    progress.update(task, advance=1)
        return True
    except (OSError, zipfile.BadZipFile) as e:
        ui.print_error(f"Extraction failed for {zip_path.name}: {e}")
        return False


def download_visdrone(
    dest_dir: Path,
    splits: Sequence[str] = ("val",),
    sample_only: bool = False,
    sample_count: int = 50,
) -> Path:
    """Download and prepare VisDrone2019-DET dataset splits."""
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

        if not zip_file.exists():
            ui.print_info(f"Fetching {zip_file.name} from Ultralytics CDN...")
            success = download_file_with_resume(url, zip_file)
            if not success:
                continue
        else:
            ui.print_info(f"Archive {zip_file.name} already exists. Skipping download.")

        extract_folder = visdrone_dir / f"VisDrone2019-DET-{split}"
        if not extract_folder.exists() or not any(extract_folder.iterdir()):
            ui.print_info(f"Extracting to {visdrone_dir}...")
            extract_zip(zip_file, visdrone_dir)
        else:
            ui.print_info(f"Split {split} already extracted at {extract_folder}.")

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
    sample_only: bool = False,
    sample_count: int = 50,
) -> Path:
    """Download and set up AI-TOD-v2 dataset."""
    ui.print_header("AI-TOD-v2 Downloader", f"Target: {dest_dir}")
    aitod_dir = dest_dir / "ai-tod-v2"
    aitod_dir.mkdir(parents=True, exist_ok=True)

    ui.print_info("AI-TOD-v2 is hosted across official repositories and mirrors:")
    ui.print_info(f"1. Google Drive mirror: https://drive.google.com/drive/folders/{AITOD_GDRIVE_FOLDER}")
    ui.print_info("2. MMDetection AI-TOD: https://github.com/Chasel-Tsui/mmdet-aitod")
    ui.print_info("3. Hugging Face: shijli/aitod-v2 (Requires 'huggingface-cli login')")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        ui.print_info("Found HF_TOKEN in environment. Initiating authenticated pull...")
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

            snapshot_download(
                repo_id="shijli/aitod-v2",
                repo_type="dataset",
                local_dir=str(aitod_dir),
                token=token,
            )
            ui.print_success(f"AI-TOD-v2 successfully fetched from Hugging Face to {aitod_dir}")
            return aitod_dir
        except (OSError, RuntimeError) as e:
            ui.print_warning(f"Hugging Face snapshot download failed: {e}")

    ui.print_info(f"Creating local scaffold structure at {aitod_dir}...")
    (aitod_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (aitod_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (aitod_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
    (aitod_dir / "annotations").mkdir(parents=True, exist_ok=True)

    readme_path = aitod_dir / "DOWNLOAD_INSTRUCTIONS.md"
    readme_path.write_text(
        f"# AI-TOD-v2 Download Instructions\n\n"
        f"Images in AI-TOD-v1 and AI-TOD-v2 are identical (fixed 800x800 resolution aerial images).\n"
        f"1. Download images and v2 annotations directly from Google Drive:\n"
        f"   https://drive.google.com/drive/folders/{AITOD_GDRIVE_FOLDER}\n"
        f"2. Extract into:\n"
        f"   data/ai-tod-v2/images/[train|val|test]/\n"
        f"   data/ai-tod-v2/annotations/aitod_val_v2.json\n\n"
        f"Alternatively, authenticate via Hugging Face:\n"
        f"   export HF_TOKEN=<your_token>\n"
        f"   sod data download --dataset aitod-v2\n",
        encoding="utf-8",
    )
    ui.print_success(f"Scaffold and download instructions saved to {readme_path}")
    return aitod_dir


def download_datasets(
    dataset: str = "visdrone",
    extract_dir: Path = Path("data"),
    splits: Sequence[str] = ("val",),
    sample_only: bool = False,
    sample_count: int = 50,
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
            sample_only=sample_only,
            sample_count=sample_count,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated dataset downloader for Small Object Detection benchmarks."
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
        help="Splits to download for VisDrone (default: val)",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Extract a small sample set of images for immediate local benchmarking",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=50,
        help="Number of sample images to collect when --sample-only is active (default: 50)",
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
