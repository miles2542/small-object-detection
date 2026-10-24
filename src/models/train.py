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

DEFAULT_WEIGHTS_DIR = Path("weights/pretrained")


def resolve_model_weights(model_identifier: str) -> str:
    """Resolve model weights file from identifier, local weights dir, or download name."""
    clean_id = model_identifier.lower().strip()
    target_filename = SUPPORTED_MODELS.get(clean_id, f"{clean_id}.pt" if not clean_id.endswith(".pt") else clean_id)

    # 1. Check local pretrained weights directory
    local_in_dir = DEFAULT_WEIGHTS_DIR / target_filename
    if local_in_dir.exists():
        return str(local_in_dir)

    # 2. Check direct path or root
    direct_path = Path(model_identifier)
    if direct_path.exists():
        return str(direct_path)

    in_root = Path(target_filename)
    if in_root.exists():
        return str(in_root)

    # 3. Return target filename so Ultralytics automatically downloads from official CDN
    return target_filename


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
) -> dict[str, Any]:
    """Execute training run for a baseline model on configured dataset."""
    weights_path = resolve_model_weights(model)
    config_path = Path(config)

    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found at: {config_path}")

    ui.print_header(
        "Baseline Model Training",
        f"Model: {model} ({weights_path}) | Config: {config_path} | Device: {device} | Epochs: {epochs}",
    )

    actual_save_dir = (
        Path(save_dir) if save_dir else Path(f"runs/train/{model.replace('.pt', '')}_aitod_mini")
    )
    actual_save_dir.parent.mkdir(parents=True, exist_ok=True)

    yolo_model = YOLO(weights_path)

    results = yolo_model.train(
        data=str(config_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=str(actual_save_dir.parent),
        name=actual_save_dir.name,
        workers=workers,
        fraction=fraction,
        plots=True,
        exist_ok=True,
        verbose=True,
    )

    best_weights = actual_save_dir / "weights" / "best.pt"
    last_weights = actual_save_dir / "weights" / "last.pt"

    ui.print_success(f"Training run completed. Checkpoints stored at {actual_save_dir / 'weights'}")

    summary: dict[str, Any] = {
        "model": model,
        "config": str(config_path),
        "epochs_completed": epochs,
        "batch_size": batch_size,
        "resolution": f"{img_size}x{img_size}",
        "device": device,
        "best_checkpoint": str(best_weights) if best_weights.exists() else None,
        "last_checkpoint": str(last_weights) if last_weights.exists() else None,
        "map50": float(getattr(results, "results_dict", {}).get("metrics/mAP50(B)", 0.0)),
        "map50_95": float(getattr(results, "results_dict", {}).get("metrics/mAP50-95(B)", 0.0)),
    }

    ui.print_table(
        title="Training Run Summary",
        columns=["Parameter / Metric", "Value"],
        rows=[
            ["Model", summary["model"]],
            ["Config", summary["config"]],
            ["Epochs Completed", str(summary["epochs_completed"])],
            ["Batch Size", str(summary["batch_size"])],
            ["Resolution", summary["resolution"]],
            ["Device", summary["device"]],
            ["Best Checkpoint", summary["best_checkpoint"] or "N/A"],
            ["Last Checkpoint", summary["last_checkpoint"] or "N/A"],
            ["mAP50(B)", f"{summary['map50']:.4f}"],
            ["mAP50-95(B)", f"{summary['map50_95']:.4f}"],
        ],
    )

    return summary
