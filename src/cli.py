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

data_app = typer.Typer(help="Dataset management, conversion, EDA, and validation.")
app.add_typer(data_app, name="data")

train_app = typer.Typer(help="Training execution across baselines and ablations.")
app.add_typer(train_app, name="train")

bench_app = typer.Typer(help="Runtime latency, throughput, and VRAM benchmarking.")
app.add_typer(bench_app, name="bench")

kaggle_app = typer.Typer(help="Headless Kaggle GPU training orchestration.")
bench_app.add_typer(kaggle_app, name="kaggle")

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
        Path(".scratch/benchmarks"),
        "--output-dir",
        "-o",
        help="Directory to save raw benchmark JSON",
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
    from src.bench.runtime import run_benchmarks

    models_list = models or ["yolov8s", "yolo26s"]
    batches_list = batch_sizes or [1, 4, 8, 16]

    run_benchmarks(
        models=models_list,
        batch_sizes=batches_list,
        img_size=img_size,
        output_dir=output_dir,
        device=device,
    )


@kaggle_app.command("scaffold")
def kaggle_scaffold(
    kernel_dir: Path = typer.Option(
        Path("kaggle/kernels/aitod_train"),
        "--kernel-dir",
        "-k",
        help="Directory to store Kaggle kernel configuration and runner script",
    ),
    kernel_id: str = typer.Option(
        "miles2542/aitod-baseline-training",
        "--kernel-id",
        help="Kaggle kernel identifier (username/slug)",
    ),
    enable_gpu: bool = typer.Option(
        True,
        "--gpu/--no-gpu",
        help="Target Dual NVIDIA T4 GPU accelerator",
    ),
    epochs: int = typer.Option(
        10,
        "--epochs",
        help="Training epochs for remote execution",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        help="Batch size for GPU dataloader",
    ),
) -> None:
    """Generate self-contained headless Kaggle runner scripts and metadata."""
    from src.bench.kaggle_runner import generate_kernel_scaffold

    generate_kernel_scaffold(
        kernel_dir=kernel_dir,
        kernel_id=kernel_id,
        enable_gpu=enable_gpu,
        epochs=epochs,
        batch_size=batch_size,
    )


@kaggle_app.command("push")
def kaggle_push(
    kernel_dir: Path = typer.Option(
        Path("kaggle/kernels/aitod_train"),
        "--kernel-dir",
        "-k",
        help="Directory containing kernel-metadata.json and runner.py",
    ),
) -> None:
    """Push kernel job to Kaggle to trigger remote Dual T4 training."""
    from src.bench.kaggle_runner import push_kernel

    push_kernel(kernel_dir=kernel_dir)


@kaggle_app.command("status")
def kaggle_status(
    kernel_id: str = typer.Option(
        "miles2542/aitod-baseline-training",
        "--kernel-id",
        help="Kaggle kernel identifier",
    ),
) -> None:
    """Check status of a running or queued Kaggle kernel."""
    from src.bench.kaggle_runner import get_kernel_status

    get_kernel_status(kernel_id=kernel_id)


@kaggle_app.command("output")
def kaggle_output(
    kernel_id: str = typer.Option(
        "miles2542/aitod-baseline-training",
        "--kernel-id",
        help="Kaggle kernel identifier",
    ),
    output_dir: Path = typer.Option(
        Path("runs/kaggle"),
        "--output-dir",
        "-o",
        help="Local directory to store downloaded weights and logs",
    ),
) -> None:
    """Download weights and output logs from completed Kaggle kernel."""
    from src.bench.kaggle_runner import download_kernel_output

    download_kernel_output(kernel_id=kernel_id, output_dir=output_dir)


@train_app.command("run")
def train_run(
    model: str = typer.Option(
        "yolov8s",
        "--model",
        "-m",
        help="Model architecture (yolov8s, yolo26s) or weights path",
    ),
    config: Path = typer.Option(
        Path("configs/aitod_full.yaml") if Path("configs/aitod_full.yaml").exists() else Path("configs/aitod_mini.yaml"),
        "--config",
        "-c",
        help="Path to dataset YAML configuration (defaults to full if present)",
    ),
    epochs: int = typer.Option(
        10,
        "--epochs",
        "-e",
        help="Number of training epochs",
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Batch size for dataloader (auto-defaults to 16 on GPU, 4 on CPU)",
    ),
    img_size: int = typer.Option(
        800,
        "--img-size",
        "-s",
        help="Input image resolution in pixels (AI-TOD standard: 800)",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        "-d",
        help="Execution device (auto-detects CUDA GPU if available, else CPU)",
    ),
    save_dir: Path | None = typer.Option(
        None,
        "--save-dir",
        help="Destination directory for checkpoints and artifacts",
    ),
    fraction: float = typer.Option(
        1.0,
        "--fraction",
        help="Fraction of dataset to train on (e.g. 0.01 for fast sanity check)",
    ),
    workers: int = typer.Option(
        2,
        "--workers",
        help="Number of dataloader workers (default: 2)",
    ),
) -> None:
    """Train baseline detection model (YOLOv8s or YOLO26s) on dataset configuration."""
    from src.models.train import train_baseline

    train_baseline(
        model=model,
        config=config,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        device=device,
        save_dir=save_dir,
        fraction=fraction,
        workers=workers,
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


@data_app.command("download")
def data_download(
    dataset: str = typer.Option(
        "visdrone",
        "--dataset",
        help="Dataset to download (visdrone, aitod-v2, all)",
    ),
    extract_dir: Path = typer.Option(
        Path("data"),
        "--extract-dir",
        help="Root directory for dataset storage",
    ),
    splits: list[str] | None = typer.Option(
        None,
        "--splits",
        help="Splits to download (e.g. val train test-dev)",
    ),
    sample_only: bool = typer.Option(
        False,
        "--sample-only",
        help="Extract a small sample set of images for immediate benchmarking",
    ),
    sample_count: int = typer.Option(
        500,
        "--sample-count",
        help="Number of sample images to collect when --sample-only is active",
    ),
) -> None:
    """Download and extract benchmark datasets (VisDrone, AI-TOD-v2)."""
    from src.data.download import download_datasets

    download_datasets(
        dataset=dataset,
        extract_dir=extract_dir,
        splits=splits or ["val"],
        sample_only=sample_only,
        sample_count=sample_count,
    )


@data_app.command("eda")
def data_eda(
    data_dir: Path = typer.Option(
        Path("data/ai-tod-v2/annotations"),
        "--data-dir",
        help="Path to annotation files directory",
    ),
    figures_dir: Path = typer.Option(
        Path("assets/eda"),
        "--figures-dir",
        help="Output directory for diagnostic plots",
    ),
    report_path: Path = typer.Option(
        Path("docs/dataset_analysis.md"),
        "--report-path",
        help="Output markdown report file path",
    ),
    summary_path: Path = typer.Option(
        Path(".scratch/eda/aitod_eda_summary.json"),
        "--summary-path",
        help="Output JSON summary file path",
    ),
) -> None:
    """Run exploratory data analysis, generate scale diagnostics and report."""
    from src.data.eda import run_eda

    run_eda(
        data_dir=data_dir,
        figures_dir=figures_dir,
        report_path=report_path,
        summary_path=summary_path,
    )


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
