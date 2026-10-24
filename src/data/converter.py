"""COCO-to-YOLO annotation converter and dataset preparation for AI-TOD-v2."""

from __future__ import annotations

import json
import logging
import math
import shutil
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.data.subset import AITOD_CLASSES, CLASS_TO_ID, select_stratified_subset

logger = logging.getLogger(__name__)


def sanitize_bbox(
    bbox: Sequence[float],
    img_w: int = 800,
    img_h: int = 800,
) -> tuple[float, float, float, float] | None:
    """Validate, sanitize, and convert a COCO bbox [x, y, w, h] to normalized YOLO format.

    Sanitization rules:
    - Drop any box where w <= 0 or h <= 0.
    - Clip box boundaries to image coordinate boundaries [0, img_w] x [0, img_h].
    - Re-check clipped width and height > 0.
    - Normalize to [x_center, y_center, w_norm, h_norm] relative to image dimensions.
    - Ensure coordinates are bounded in [0.0, 1.0] without NaN or Inf.

    Args:
        bbox: Sequence of [x, y, w, h] coordinates.
        img_w: Image width in pixels (default: 800).
        img_h: Image height in pixels (default: 800).

    Returns:
        tuple of (x_center, y_center, w_norm, h_norm) normalized in [0, 1], or None if invalid.
    """
    if len(bbox) < 4:
        return None

    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    # Explicit check for zero or negative dimensions
    if w <= 0.0 or h <= 0.0:
        return None

    # Check for NaN / Inf in inputs
    if any(math.isnan(v) or math.isinf(v) for v in (x, y, w, h)):
        return None

    # Clip coordinates to image boundary
    x1 = max(0.0, min(x, float(img_w)))
    y1 = max(0.0, min(y, float(img_h)))
    x2 = max(0.0, min(x + w, float(img_w)))
    y2 = max(0.0, min(y + h, float(img_h)))

    clipped_w = x2 - x1
    clipped_h = y2 - y1

    if clipped_w <= 0.0 or clipped_h <= 0.0:
        return None

    # Compute normalized center coordinates and dimensions
    x_center = (x1 + clipped_w / 2.0) / float(img_w)
    y_center = (y1 + clipped_h / 2.0) / float(img_h)
    w_norm = clipped_w / float(img_w)
    h_norm = clipped_h / float(img_h)

    # Clamp bounds strictly to [0.0, 1.0]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return x_center, y_center, w_norm, h_norm


def create_placeholder_image(output_path: Path, width: int = 800, height: int = 800) -> None:
    """Create a minimal valid 800x800 PNG file when source aerial image is absent."""
    try:
        from PIL import Image

        img = Image.new("RGB", (width, height), color=(30, 30, 30))
        img.save(output_path, format="PNG")
    except ImportError:
        # Fallback to pure python minimal PNG header if PIL is unavailable
        import struct
        import zlib

        raw_data = b"".join(b"\x00" + b"\x1e\x1e\x1e" * width for _ in range(height))
        compressed = zlib.compress(raw_data)

        def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + chunk_type
                + data
                + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
            )

        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        png = (
            b"\x89PNG\r\n\x1a\n"
            + make_chunk(b"IHDR", ihdr_data)
            + make_chunk(b"IDAT", compressed)
            + make_chunk(b"IEND", b"")
        )
        output_path.write_bytes(png)


def convert_coco_to_yolo(
    coco_data: dict[str, Any],
    labels_dir: Path,
    image_ids: Sequence[int] | None = None,
    images_source_dir: Path | None = None,
    images_dest_dir: Path | None = None,
    img_w: int = 800,
    img_h: int = 800,
    create_placeholders_if_missing: bool = True,
) -> dict[str, Any]:
    """Convert COCO annotations to normalized YOLO format text files.

    Args:
        coco_data: Loaded COCO JSON dict.
        labels_dir: Destination directory for YOLO .txt label files.
        image_ids: Optional list of image IDs to convert. If None, converts all images.
        images_source_dir: Directory where raw images are located.
        images_dest_dir: Destination directory to copy/link images for YOLO training.
        img_w: Image width (default 800).
        img_h: Image height (default 800).
        create_placeholders_if_missing: Whether to generate placeholder PNGs when raw images are absent.

    Returns:
        Conversion statistics summary dict.
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    if images_dest_dir:
        images_dest_dir.mkdir(parents=True, exist_ok=True)

    categories = coco_data.get("categories", [])
    raw_cat_to_name = {c["id"]: c["name"] for c in categories}

    all_images = coco_data.get("images", [])
    target_id_set = set(image_ids) if image_ids is not None else None

    # Filter images
    if target_id_set is not None:
        images_to_process = [img for img in all_images if img["id"] in target_id_set]
    else:
        images_to_process = all_images

    img_id_to_file = {img["id"]: img["file_name"] for img in images_to_process}

    # Group annotations by image_id
    img_to_anns: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco_data.get("annotations", []):
        iid = ann.get("image_id")
        if iid in img_id_to_file:
            img_to_anns[iid].append(ann)

    total_annotations_seen = 0
    total_valid_converted = 0
    total_zero_dim_dropped = 0
    total_out_of_bounds_dropped = 0
    class_converted_counts: dict[str, int] = {name: 0 for name in AITOD_CLASSES}

    images_linked = 0
    images_placeholder_created = 0

    for img in images_to_process:
        iid = img["id"]
        fname = img["file_name"]
        stem = Path(fname).stem
        label_file = labels_dir / f"{stem}.txt"

        # Handle image file linking / copying / placeholder
        if images_dest_dir:
            dest_img_path = images_dest_dir / fname
            if not dest_img_path.exists():
                src_img_path = (images_source_dir / fname) if images_source_dir else None
                if src_img_path and src_img_path.exists():
                    shutil.copy2(src_img_path, dest_img_path)
                    images_linked += 1
                elif create_placeholders_if_missing:
                    create_placeholder_image(dest_img_path, width=img_w, height=img_h)
                    images_placeholder_created += 1

        anns = img_to_anns.get(iid, [])
        lines = []

        for ann in anns:
            total_annotations_seen += 1
            bbox = ann.get("bbox", [])
            raw_cid = ann.get("category_id")
            cat_name = raw_cat_to_name.get(raw_cid)

            if cat_name not in CLASS_TO_ID:
                continue

            canonical_class_id = CLASS_TO_ID[cat_name]

            # Dimension checks
            if len(bbox) < 4 or float(bbox[2]) <= 0.0 or float(bbox[3]) <= 0.0:
                total_zero_dim_dropped += 1
                continue

            sanitized = sanitize_bbox(bbox, img_w=img_w, img_h=img_h)
            if sanitized is None:
                total_out_of_bounds_dropped += 1
                continue

            xc, yc, wn, hn = sanitized
            lines.append(f"{canonical_class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
            total_valid_converted += 1
            class_converted_counts[cat_name] += 1

        # Write YOLO label file (empty file for negative images)
        with open(label_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

    stats: dict[str, Any] = {
        "images_processed": len(images_to_process),
        "labels_written": len(images_to_process),
        "total_annotations_seen": total_annotations_seen,
        "valid_converted_annotations": total_valid_converted,
        "zero_dim_annotations_dropped": total_zero_dim_dropped,
        "out_of_bounds_annotations_dropped": total_out_of_bounds_dropped,
        "class_counts": class_converted_counts,
        "images_copied": images_linked,
        "placeholders_generated": images_placeholder_created,
    }

    return stats


def generate_ultralytics_yaml(
    output_path: Path,
    dataset_root: Path,
    train_rel: str = "images/train",
    val_rel: str = "images/val",
) -> None:
    """Generate an Ultralytics-compatible dataset configuration YAML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(AITOD_CLASSES))

    yaml_content = f"""# Ultralytics dataset configuration for AI-TOD-v2
# Generated by src.data.converter

path: {dataset_root.as_posix()}
train: {train_rel}
val: {val_rel}

# AI-TOD-v2 0-indexed canonical classes
names:
{names_block}
"""
    output_path.write_text(yaml_content, encoding="utf-8")
    logger.info("Generated Ultralytics dataset config: %s", output_path)


def create_stratified_mini_dataset(
    train_json: Path,
    val_json: Path,
    source_train_imgs: Path,
    source_val_imgs: Path,
    output_root: Path,
    train_size: int = 1000,
    val_size: int = 200,
    seed: int = 42,
    config_yaml_path: Path | None = None,
    create_placeholders_if_missing: bool = True,
) -> dict[str, Any]:
    """Generate the stratified development mini dataset for training and validation.

    Outputs dataset structure:
      output_root / images / train
      output_root / images / val
      output_root / labels / train
      output_root / labels / val
    And dataset YAML:
      config_yaml_path (default: configs/aitod_mini.yaml)

    Args:
        train_json: Path to aitodv2_train.json.
        val_json: Path to aitod_val.json.
        source_train_imgs: Path to raw train images folder.
        source_val_imgs: Path to raw val images folder.
        output_root: Root directory for mini dataset (data/ai-tod-v2/splits/mini).
        train_size: Number of training images (default 1000).
        val_size: Number of validation images (default 200).
        seed: Random seed for reproducibility.
        config_yaml_path: Path to write Ultralytics YAML config.
        create_placeholders_if_missing: Whether to generate placeholder PNGs when source images are absent.

    Returns:
        Detailed summary dict containing sampling and conversion metrics.
    """
    logger.info("Loading training annotations: %s", train_json)
    with open(train_json, encoding="utf-8") as f:
        train_coco = json.load(f)

    logger.info("Loading validation annotations: %s", val_json)
    with open(val_json, encoding="utf-8") as f:
        val_coco = json.load(f)

    # 1. Stratified subset selection
    logger.info("Selecting %d stratified train images...", train_size)
    train_ids, train_strat_stats = select_stratified_subset(
        train_coco, target_size=train_size, seed=seed, rare_class_boost=True
    )

    logger.info("Selecting %d stratified val images...", val_size)
    val_ids, val_strat_stats = select_stratified_subset(
        val_coco, target_size=val_size, seed=seed, rare_class_boost=True
    )

    # 2. Convert train split
    train_images_dest = output_root / "images" / "train"
    train_labels_dest = output_root / "labels" / "train"
    logger.info("Converting train split to YOLO format in %s...", train_labels_dest)
    train_conv_stats = convert_coco_to_yolo(
        coco_data=train_coco,
        labels_dir=train_labels_dest,
        image_ids=train_ids,
        images_source_dir=source_train_imgs,
        images_dest_dir=train_images_dest,
        create_placeholders_if_missing=create_placeholders_if_missing,
    )

    # 3. Convert val split
    val_images_dest = output_root / "images" / "val"
    val_labels_dest = output_root / "labels" / "val"
    logger.info("Converting val split to YOLO format in %s...", val_labels_dest)
    val_conv_stats = convert_coco_to_yolo(
        coco_data=val_coco,
        labels_dir=val_labels_dest,
        image_ids=val_ids,
        images_source_dir=source_val_imgs,
        images_dest_dir=val_images_dest,
        create_placeholders_if_missing=create_placeholders_if_missing,
    )

    # 4. Generate Ultralytics configuration YAML
    if config_yaml_path is None:
        config_yaml_path = Path("configs/aitod_mini.yaml")

    generate_ultralytics_yaml(
        output_path=config_yaml_path,
        dataset_root=output_root,
        train_rel="images/train",
        val_rel="images/val",
    )

    summary: dict[str, Any] = {
        "dataset_name": "ai-tod-v2-mini",
        "output_root": str(output_root),
        "config_yaml": str(config_yaml_path),
        "train": {
            "stratification": train_strat_stats,
            "conversion": train_conv_stats,
        },
        "val": {
            "stratification": val_strat_stats,
            "conversion": val_conv_stats,
        },
    }

    return summary
