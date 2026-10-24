"""Modular CLI interface for the Small Object Detection benchmarking suite."""

from __future__ import annotations

from pathlib import Path

import typer

from src.utils import ui

app = typer.Typer(
    name="sod",
    help="Small Object Detection (AI-TOD-v2 / YOLOv8 / YOLO26) Benchmarking Suite.",
    add_completion=False,
)

data_app = typer.Typer(help="Dataset management, conversion, and validation.")
app.add_typer(data_app, name="data")

train_app = typer.Typer(help="Training execution across baselines and ablations.")
app.add_typer(train_app, name="train")

bench_app = typer.Typer(help="Runtime latency, throughput, and VRAM benchmarking.")
app.add_typer(bench_app, name="bench")

eval_app = typer.Typer(help="Dual-surface evaluation on AI-TOD micro-scale tiers.")
app.add_typer(eval_app, name="eval")


@app.callback()
def main_callback() -> None:
    """Entry point for Small Object Detection CLI."""


@bench_app.command("runtime")
def bench_runtime(
    config: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
    models: list[str] | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model architectures to benchmark (e.g. yolov8s, yolo26s)",
    ),
    batch_sizes: list[int] | None = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Batch sizes to benchmark (e.g. 1 4 8 16)",
    ),
    output_dir: Path = typer.Option(
        Path("benchmarks"),
        "--output-dir",
        "-o",
        help="Directory to save raw benchmark JSON and summary markdown report",
    ),
    img_size: int = typer.Option(
        800,
        "--img-size",
        "-s",
        help="Input image resolution (AI-TOD standard: 800)",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Execution device (e.g. cpu, directml, cuda:0)",
    ),
) -> None:
    """Benchmark inference latency, FPS throughput, VRAM, and gradient backward pass."""
    from scripts.benchmark_runtime import run_benchmarks

    models_list = models or ["yolov8s", "yolo26s"]
    batches_list = batch_sizes or [1, 4, 8, 16]

    run_benchmarks(
        models=models_list,
        batch_sizes=batches_list,
        img_size=img_size,
        output_dir=output_dir,
        device=device,
    )


@data_app.command("summary")
def data_summary(
    config: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
) -> None:
    """Display configured dataset paths, classes, and micro-scale bin intervals."""
    import yaml

    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg.get("dataset", {})
    bins = ds_cfg.get("micro_bins", {})
    rows = [[k, f"{v[0]} - {v[1]} px"] for k, v in bins.items()]
    ui.print_table("Micro-scale Bins (AI-TOD)", ["Bin", "Pixel Area Interval"], rows)


@data_app.command("prepare")
def data_prepare(
    train_json: Path = typer.Option(
        Path("data/ai-tod-v2/annotations/aitodv2_train.json"),
        "--train-json",
        help="Path to training COCO annotations JSON",
    ),
    val_json: Path = typer.Option(
        Path("data/ai-tod-v2/annotations/aitod_val.json"),
        "--val-json",
        help="Path to validation COCO annotations JSON",
    ),
    source_train_imgs: Path = typer.Option(
        Path("data/ai-tod-v2/images/train"),
        "--source-train-imgs",
        help="Path to directory containing raw training images",
    ),
    source_val_imgs: Path = typer.Option(
        Path("data/ai-tod-v2/images/val"),
        "--source-val-imgs",
        help="Path to directory containing raw validation images",
    ),
    output_dir: Path = typer.Option(
        Path("data/ai-tod-v2/splits/mini"),
        "--output-dir",
        "-o",
        help="Destination directory for stratified mini split",
    ),
    train_size: int = typer.Option(
        1000,
        "--train-size",
        help="Number of training images for stratified mini split",
    ),
    val_size: int = typer.Option(
        200,
        "--val-size",
        help="Number of validation images for stratified mini split",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for deterministic sampling",
    ),
    config_yaml: Path = typer.Option(
        Path("configs/aitod_mini.yaml"),
        "--config-yaml",
        help="Path to write Ultralytics dataset configuration YAML",
    ),
    placeholders: bool = typer.Option(
        True,
        "--placeholders/--no-placeholders",
        help="Create placeholder images if source aerial image files are absent",
    ),
) -> None:
    """Generate stratified development mini split and convert annotations to YOLO format."""
    from src.data.converter import create_stratified_mini_dataset

    ui.print_header(
        "AI-TOD-v2 Stratified Development Subset & YOLO Converter",
        f"Output: {output_dir} | Train: {train_size} imgs | Val: {val_size} imgs",
    )

    if not train_json.exists():
        ui.print_error(f"Training annotation file not found: {train_json}")
        raise typer.Exit(code=1)

    if not val_json.exists():
        ui.print_error(f"Validation annotation file not found: {val_json}")
        raise typer.Exit(code=1)

    summary = create_stratified_mini_dataset(
        train_json=train_json,
        val_json=val_json,
        source_train_imgs=source_train_imgs,
        source_val_imgs=source_val_imgs,
        output_root=output_dir,
        train_size=train_size,
        val_size=val_size,
        seed=seed,
        config_yaml_path=config_yaml,
        create_placeholders_if_missing=placeholders,
    )

    train_conv = summary["train"]["conversion"]
    val_conv = summary["val"]["conversion"]

    ui.print_success(f"Generated Ultralytics dataset configuration at {config_yaml}")
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


if __name__ == "__main__":
    app()
