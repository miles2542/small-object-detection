"""Modular CLI interface for the Small Object Detection benchmarking suite."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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
    models: Optional[list[str]] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model architectures to benchmark (e.g. yolov8s, yolo26s)",
    ),
    batch_sizes: Optional[list[int]] = typer.Option(
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


if __name__ == "__main__":
    app()
