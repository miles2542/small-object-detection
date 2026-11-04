"""Standard high-resolution inference pipeline for full-image scaling benchmarks."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from PIL import Image
from ultralytics import YOLO

from src.eval.audit import sha256
from src.eval.runner import resolve_inputs
from src.infer.sahi import (
    CANONICAL_CLASSES,
    build_category_mapping,
)


@dataclass
class HighResConfig:
    """Configuration parameters for high-resolution full-image inference."""

    image_size: int = 800
    confidence_threshold: float = 0.001
    iou_threshold: float = 0.70
    max_detections_per_image: int = 1500
    class_agnostic: bool = False
    batch_size: int = 1

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in [0.0, 1.0]")
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError("iou_threshold must be in [0.0, 1.0]")
        if self.max_detections_per_image <= 0:
            raise ValueError("max_detections_per_image must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


def get_peak_memory_mb(device: str = "cpu") -> float:
    """Retrieve peak memory consumption in megabytes (GPU VRAM or CPU RAM)."""
    if ("cuda" in device or device.isdigit()) and torch.cuda.is_available():
        dev_idx = int(device) if device.isdigit() else torch.cuda.current_device()
        return float(torch.cuda.max_memory_allocated(dev_idx) / (1024 * 1024))
    process = psutil.Process()
    return float(process.memory_info().rss / (1024 * 1024))


def reset_peak_memory(device: str = "cpu") -> None:
    """Reset peak memory tracking counters if supported on the device."""
    if ("cuda" in device or device.isdigit()) and torch.cuda.is_available():
        dev_idx = int(device) if device.isdigit() else torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(dev_idx)


class HighResInferenceRunner:
    """Executes full-frame forward inference on scaled inputs and logs runtime metrics."""

    def __init__(
        self,
        weights: str | Path | None = None,
        model: Any = None,
        device: str = "cpu",
        config: HighResConfig | None = None,
        category_map: dict[int, int] | None = None,
        smoke: bool = False,
    ) -> None:
        self.config = config or HighResConfig()
        self.device = device
        self.smoke = smoke

        if model is not None:
            self.model = model
        else:
            if weights is None:
                raise ValueError("Must provide either weights path or pre-initialized model")
            weights_path = Path(weights)
            if not weights_path.is_file():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            self.model = YOLO(str(weights_path), task="detect")

        raw_names = getattr(self.model, "names", None)
        if raw_names is None:
            raw_names = CANONICAL_CLASSES

        if category_map is not None:
            self.category_map = category_map
        else:
            self.category_map = build_category_mapping(raw_names, smoke=self.smoke)

    def predict_image(
        self,
        image: str | Path | np.ndarray | Image.Image,
        image_id: int = 0,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Run inference on a single image and return standardized COCO predictions.

        Bounding boxes are converted to [x_min, y_min, width, height] in original pixel coordinates.
        """
        reset_peak_memory(self.device)
        start_time = time.perf_counter()

        # Handle native end-to-end NMS models if applicable
        head = getattr(getattr(self.model, "model", None), "model", None)
        native_end2end = False
        if head and hasattr(head, "__getitem__"):
            last_layer = head[-1]
            native_end2end = getattr(last_layer, "one2one_cv2", None) is not None

        predict_kwargs: dict[str, Any] = {
            "source": image if isinstance(image, (np.ndarray, Image.Image)) else str(image),
            "imgsz": self.config.image_size,
            "conf": self.config.confidence_threshold,
            "iou": self.config.iou_threshold,
            "max_det": self.config.max_detections_per_image,
            "agnostic_nms": self.config.class_agnostic,
            "device": self.device,
            "verbose": False,
            "save": False,
        }
        if native_end2end:
            predict_kwargs["nms"] = False

        results = self.model.predict(**predict_kwargs)
        result = results[0] if isinstance(results, list) else next(iter(results))

        total_wall_ms = (time.perf_counter() - start_time) * 1000.0

        # Extract timing from predictor speeds if available
        speed = getattr(result, "speed", {}) or {}
        preprocess_ms = float(speed.get("preprocess", 0.0))
        inference_ms = float(speed.get("inference", 0.0))
        postprocess_ms = float(speed.get("postprocess", 0.0))

        if preprocess_ms == 0.0 and inference_ms == 0.0 and postprocess_ms == 0.0:
            inference_ms = total_wall_ms

        peak_mem_mb = get_peak_memory_mb(self.device)

        durations = {
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": total_wall_ms,
            "peak_memory_mb": round(peak_mem_mb, 2),
        }

        detections: list[dict[str, Any]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()

            for box, score, cls_idx in zip(boxes_xyxy, confs, classes, strict=True):
                cls_int = int(cls_idx)
                if cls_int not in self.category_map:
                    continue

                target_cat_id = self.category_map[cls_int]
                score_val = float(score)
                if not (0.0 <= score_val <= 1.0) or not math.isfinite(score_val):
                    continue

                x1, y1, x2, y2 = (float(v) for v in box)
                if not all(math.isfinite(v) for v in [x1, y1, x2, y2]):
                    continue

                width = x2 - x1
                height = y2 - y1
                if width <= 0 or height <= 0:
                    continue

                detections.append(
                    {
                        "image_id": image_id,
                        "category_id": target_cat_id,
                        "bbox": [x1, y1, width, height],
                        "score": score_val,
                    }
                )

        detections.sort(key=lambda d: d["score"], reverse=True)
        if len(detections) > self.config.max_detections_per_image:
            detections = detections[: self.config.max_detections_per_image]

        return detections, durations


def export_high_res_predictions(
    weights: Path,
    data: Path,
    output_dir: Path,
    *,
    split: str = "val",
    image_size: int = 800,
    config: HighResConfig | None = None,
    annotations: Path | None = None,
    image_root: Path | None = None,
    image_ids: list[int] | None = None,
    device: str = "cpu",
    smoke: bool = False,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Execute high-resolution full-image inference and export predictions and manifest."""
    weights = Path(weights)
    data = Path(data)
    output_dir = Path(output_dir)

    cfg = config or HighResConfig(image_size=image_size)
    if cfg.image_size != image_size:
        cfg.image_size = image_size

    resolved_ann, resolved_root, resolved_ids = resolve_inputs(
        data=data,
        split=split,
        annotations=annotations,
        image_root=image_root,
    )

    if resolved_root is None:
        raise ValueError("Could not determine image root directory for evaluation split")

    coco_gt = json.loads(resolved_ann.read_text(encoding="utf-8"))
    images_by_id = {img["id"]: img for img in coco_gt["images"]}

    selected_ids = resolved_ids if image_ids is None else image_ids
    if sample_limit is not None:
        selected_ids = selected_ids[:sample_limit]
    if not selected_ids:
        raise ValueError("No images selected for inference")

    runner = HighResInferenceRunner(
        weights=weights,
        device=device,
        config=cfg,
        smoke=smoke,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.json"
    manifest_path = output_dir / "manifest.json"

    all_predictions: list[dict[str, Any]] = []
    timing_records: list[dict[str, float]] = []
    detection_counts: dict[str, int] = {}

    for iid in selected_ids:
        img_info = images_by_id[iid]
        img_file = resolved_root / img_info["file_name"]
        if not img_file.is_file():
            raise FileNotFoundError(f"Image file not found: {img_file}")

        preds, timing = runner.predict_image(image=img_file, image_id=iid)
        all_predictions.extend(preds)
        timing_records.append(timing)
        detection_counts[str(iid)] = len(preds)

    pred_path.write_text(json.dumps(all_predictions, indent=2), encoding="utf-8")

    mean_total_ms = (
        float(np.mean([t["total_ms"] for t in timing_records])) if timing_records else 0.0
    )
    mean_prep_ms = (
        float(np.mean([t["preprocess_ms"] for t in timing_records])) if timing_records else 0.0
    )
    mean_infer_ms = (
        float(np.mean([t["inference_ms"] for t in timing_records])) if timing_records else 0.0
    )
    mean_post_ms = (
        float(np.mean([t["postprocess_ms"] for t in timing_records])) if timing_records else 0.0
    )
    max_peak_mem = (
        float(np.max([t["peak_memory_mb"] for t in timing_records])) if timing_records else 0.0
    )

    manifest = {
        "status": "complete",
        "purpose": "smoke_only" if smoke else "evaluation",
        "strategy": "high_res_standard",
        "image_size": cfg.image_size,
        "weights": {"path": weights.as_posix(), "sha256": sha256(weights)},
        "annotations": {"path": resolved_ann.as_posix(), "sha256": sha256(resolved_ann)},
        "image_root": resolved_root.as_posix(),
        "image_ids": selected_ids,
        "images": [{"id": iid, "file_name": images_by_id[iid]["file_name"]} for iid in selected_ids],
        "category_mapping": {
            str(i): {"name": runner.category_map.get(i, ""), "category_id": cid}
            for i, cid in runner.category_map.items()
        },
        "prediction_counts": detection_counts,
        "detections": len(all_predictions),
        "zero_detection_image_ids": [iid for iid in selected_ids if detection_counts[str(iid)] == 0],
        "cap_hit_image_ids": [
            iid
            for iid in selected_ids
            if detection_counts[str(iid)] == cfg.max_detections_per_image
        ],
        "predict_settings": {
            "imgsz": cfg.image_size,
            "conf": cfg.confidence_threshold,
            "iou": cfg.iou_threshold,
            "max_det": cfg.max_detections_per_image,
            "agnostic_nms": cfg.class_agnostic,
            "augment": False,
            "rect": False,
        },
        "config": asdict(cfg),
        "total_images": len(selected_ids),
        "total_detections": len(all_predictions),
        "latency": {
            "mean_preprocess_ms": round(mean_prep_ms, 2),
            "mean_inference_ms": round(mean_infer_ms, 2),
            "mean_postprocess_ms": round(mean_post_ms, 2),
            "mean_total_ms": round(mean_total_ms, 2),
            "throughput_fps": round(1000.0 / mean_total_ms, 2) if mean_total_ms > 0 else 0.0,
        },
        "memory": {
            "peak_mb": round(max_peak_mem, 2),
        },
        "predictions_sha256": sha256(pred_path),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def benchmark_resolutions(
    weights: Path,
    data: Path,
    output_dir: Path,
    *,
    resolutions: list[int] | None = None,
    split: str = "val",
    annotations: Path | None = None,
    image_root: Path | None = None,
    device: str = "cpu",
    smoke: bool = False,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Benchmark inference latency, memory, and throughput across multiple image resolutions."""
    resolutions = resolutions or [800, 1280, 1600]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-resolve image IDs to benchmark on identical samples across resolutions
    _resolved_ann, _resolved_root, resolved_ids = resolve_inputs(
        data=Path(data),
        split=split,
        annotations=annotations,
        image_root=image_root,
    )
    if not resolved_ids:
        raise ValueError("No images resolved from data configuration")

    test_ids = resolved_ids[:sample_limit] if sample_limit is not None else resolved_ids

    benchmark_summary: dict[str, Any] = {
        "weights": str(weights),
        "device": device,
        "sample_count": len(test_ids),
        "resolutions": {},
    }

    for res in resolutions:
        sub_dir = output_dir / f"res_{res}"
        manifest = export_high_res_predictions(
            weights=weights,
            data=data,
            output_dir=sub_dir,
            split=split,
            image_size=res,
            annotations=annotations,
            image_root=image_root,
            image_ids=test_ids,
            device=device,
            smoke=smoke,
        )
        benchmark_summary["resolutions"][str(res)] = {
            "image_size": res,
            "latency": manifest["latency"],
            "memory": manifest["memory"],
            "total_detections": manifest["total_detections"],
        }

    summary_file = output_dir / "resolution_benchmark.json"
    summary_file.write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    return benchmark_summary
