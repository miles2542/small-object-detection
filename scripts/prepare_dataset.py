"""Prepare stratified development split and convert annotations to YOLO format."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.converter import create_stratified_mini_dataset
from src.utils import ui


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate stratified AI-TOD-v2 mini split and convert to YOLO format."
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=Path("data/ai-tod-v2/annotations/aitodv2_train.json"),
        help="Path to training COCO annotations JSON.",
    )
    parser.add_argument(
        "--val-json",
        type=Path,
        default=Path("data/ai-tod-v2/annotations/aitod_val.json"),
        help="Path to validation COCO annotations JSON.",
    )
    parser.add_argument(
        "--source-train-imgs",
        type=Path,
        default=Path("data/ai-tod-v2/images/train"),
        help="Directory with raw training images.",
    )
    parser.add_argument(
        "--source-val-imgs",
        type=Path,
        default=Path("data/ai-tod-v2/images/val"),
        help="Directory with raw validation images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ai-tod-v2/splits/mini"),
        help="Output directory for stratified mini split.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=1000,
        help="Number of training images for mini split (default: 1000).",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=200,
        help="Number of validation images for mini split (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42).",
    )
    parser.add_argument(
        "--config-yaml",
        type=Path,
        default=Path("configs/aitod_mini.yaml"),
        help="Output path for Ultralytics dataset YAML config (default: configs/aitod_mini.yaml).",
    )
    parser.add_argument(
        "--no-placeholders",
        action="store_true",
        help="Disable automatic placeholder generation when raw images are missing.",
    )

    args = parser.parse_args()

    ui.print_header(
        "AI-TOD-v2 Stratified Development Subset & YOLO Converter",
        f"Output: {args.output_dir} | Train: {args.train_size} imgs | Val: {args.val_size} imgs",
    )

    if not args.train_json.exists():
        ui.print_error(f"Training annotation file not found: {args.train_json}")
        return

    if not args.val_json.exists():
        ui.print_error(f"Validation annotation file not found: {args.val_json}")
        return

    summary = create_stratified_mini_dataset(
        train_json=args.train_json,
        val_json=args.val_json,
        source_train_imgs=args.source_train_imgs,
        source_val_imgs=args.source_val_imgs,
        output_root=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        config_yaml_path=args.config_yaml,
        create_placeholders_if_missing=not args.no_placeholders,
    )

    # Render results table
    train_conv = summary["train"]["conversion"]
    val_conv = summary["val"]["conversion"]

    ui.print_success(f"Generated Ultralytics dataset configuration at {args.config_yaml}")
    ui.print_table(
        "Conversion and Sanitization Summary",
        ["Metric", "Train Split", "Val Split"],
        [
            ["Images Processed", str(train_conv["images_processed"]), str(val_conv["images_processed"])],
            ["Labels Written", str(train_conv["labels_written"]), str(val_conv["labels_written"])],
            ["Total Anns Parsed", str(train_conv["total_annotations_seen"]), str(val_conv["total_annotations_seen"])],
            ["Valid Boxes Converted", str(train_conv["valid_converted_annotations"]), str(val_conv["valid_converted_annotations"])],
            ["Zero-dim Boxes Dropped", str(train_conv["zero_dim_annotations_dropped"]), str(val_conv["zero_dim_annotations_dropped"])],
            ["Out-of-bounds Dropped", str(train_conv["out_of_bounds_annotations_dropped"]), str(val_conv["out_of_bounds_annotations_dropped"])],
        ],
    )

    # Render class distribution table
    train_strat = summary["train"]["stratification"]
    cat_rows = []
    for cname, cdata in train_strat["category_comparison"].items():
        cat_rows.append([
            cname,
            f"{cdata['subset_count']:,}",
            f"{cdata['subset_pct']:.2f}%",
            f"{cdata['population_count']:,}",
            f"{cdata['population_pct']:.2f}%",
        ])
    ui.print_table(
        "Train Split Class Representation (Subset vs Population)",
        ["Category", "Subset Count", "Subset %", "Pop Count", "Pop %"],
        cat_rows,
    )

    # Render scale bin distribution table
    scale_rows = []
    for sbin, sdata in train_strat["scale_bin_comparison"].items():
        scale_rows.append([
            sbin,
            f"{sdata['subset_count']:,}",
            f"{sdata['subset_pct']:.2f}%",
            f"{sdata['population_count']:,}",
            f"{sdata['population_pct']:.2f}%",
        ])
    ui.print_table(
        "Train Split Micro-scale Bins (Subset vs Population)",
        ["Scale Bin", "Subset Count", "Subset %", "Pop Count", "Pop %"],
        scale_rows,
    )


if __name__ == "__main__":
    main()
