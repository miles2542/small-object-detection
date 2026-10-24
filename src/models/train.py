"""Modular baseline training pipeline for YOLOv8s and YOLO26s on AI-TOD-v2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.utils import ui

logger = logging.getLogger(__name__)

SUPPORTED_MODELS: dict[str, str] = {
    "yolov8s": "yolov8s.pt",
    "yolo26s": "yolo26s.pt",
}


def resolve_model_weights(model_identifier: str) -> str:
    """Resolve model weights file from identifier or path."""
    clean_id = model_identifier.lower().strip()
    if clean_id in SUPPORTED_MODELS:
        weights_name = SUPPORTED_MODELS[clean_id]
        if Path(weights_name).exists():
            return weights_name
        return weights_name

    p = Path(model_identifier)
    if p.exists():
        return str(p)

    candidate = Path(f"{model_identifier}.pt")
    if candidate.exists():
        return str(candidate)

    return model_identifier


def train_baseline(
    model: str = "yolov8s",
    config: Path | str = "configs/aitod_mini.yaml",
    epochs: int = 1,
    batch_size: int = 2,
    img_size: int = 800,
    device: str = "cpu",
    save_dir: Path | str | None = None,
    fraction: float = 1.0,
    workers: int = 0,
    seed: int = 42,
    val: bool = True,
    exist_ok: bool = True,
) -> dict[str, Any]:
    """Execute baseline model training on dataset configuration.

    Args:
        model: Model architecture name (e.g. 'yolov8s', 'yolo26s') or weights path.
        config: Path to dataset YAML configuration file.
        epochs: Number of training epochs.
        batch_size: Batch size for dataloader.
        img_size: Input image resolution in pixels (default: 800 for AI-TOD standard).
        device: Computation device ('cpu', '0', 'cuda:0', etc.).
        save_dir: Destination directory for training artifacts and checkpoints.
        fraction: Fraction of dataset to train on (default 1.0; e.g. 0.01 for sanity runs).
        workers: Number of dataloader worker processes (default: 0 for CPU/Windows safety).
        seed: Random seed for deterministic reproducibility.
        val: Whether to run validation after training.
        exist_ok: Whether to overwrite existing run directory.

    Returns:
        Structured dictionary containing training metrics, checkpoint paths, and run config.
    """
    config_path = Path(config).resolve()
    if not config_path.exists():
        msg = f"Dataset configuration YAML not found at: {config_path}"
        ui.print_error(msg)
        raise FileNotFoundError(msg)

    weights_target = resolve_model_weights(model)
    ui.print_header(
        "Baseline Model Training Pipeline",
        f"Model: {model} ({weights_target}) | Config: {config_path.name} | Device: {device}",
    )

    if save_dir:
        resolved_save_dir = Path(save_dir).resolve()
        project_dir = str(resolved_save_dir.parent)
        run_name = resolved_save_dir.name
    else:
        project_dir = str(Path("runs/train").resolve())
        run_name = f"{model}_{config_path.stem}"

    ui.print_info(f"Loading model weights from {weights_target}...")
    yolo_model = YOLO(weights_target)

    ui.print_info(
        f"Beginning training: {epochs} epoch(s), batch {batch_size}, "
        f"imgsz {img_size}, fraction {fraction}, device {device}"
    )

    train_results = yolo_model.train(
        data=str(config_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project_dir,
        name=run_name,
        fraction=fraction,
        workers=workers,
        seed=seed,
        val=val,
        exist_ok=exist_ok,
        verbose=True,
    )

    actual_save_dir = Path(project_dir) / run_name
    best_weights = actual_save_dir / "weights" / "best.pt"
    last_weights = actual_save_dir / "weights" / "last.pt"

    # Extract metrics safely from Ultralytics training results
    metrics_summary: dict[str, Any] = {}
    if hasattr(train_results, "results_dict") and isinstance(train_results.results_dict, dict):
        metrics_summary = train_results.results_dict
    elif hasattr(train_results, "box") and train_results.box is not None:
        metrics_summary = {
            "metrics/mAP50(B)": getattr(train_results.box, "map50", 0.0),
            "metrics/mAP50-95(B)": getattr(train_results.box, "map", 0.0),
        }

    summary: dict[str, Any] = {
        "model": model,
        "weights": weights_target,
        "config": str(config_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "device": device,
        "save_dir": str(actual_save_dir),
        "best_weights": str(best_weights) if best_weights.exists() else None,
        "last_weights": str(last_weights) if last_weights.exists() else None,
        "metrics": metrics_summary,
    }

    ui.print_success(f"Training run completed. Checkpoints stored at {actual_save_dir / 'weights'}")
    ui.print_table(
        "Training Run Summary",
        ["Parameter / Metric", "Value"],
        [
            ["Model", model],
            ["Config", config_path.name],
            ["Epochs Completed", str(epochs)],
            ["Batch Size", str(batch_size)],
            ["Resolution", f"{img_size}x{img_size}"],
            ["Device", device],
            ["Best Checkpoint", str(best_weights) if best_weights.exists() else "None"],
            ["Last Checkpoint", str(last_weights) if last_weights.exists() else "None"],
            ["mAP50(B)", f"{metrics_summary.get('metrics/mAP50(B)', 0.0):.4f}"],
            ["mAP50-95(B)", f"{metrics_summary.get('metrics/mAP50-95(B)', 0.0):.4f}"],
        ],
    )

    return summary
