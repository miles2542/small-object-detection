"""Core SAHI slicing inference engine for micro-scale aerial object detection."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from src.eval.audit import sha256
from src.eval.runner import resolve_inputs

CANONICAL_CLASSES: dict[int, str] = {
    0: "airplane",
    1: "bridge",
    2: "storage-tank",
    3: "ship",
    4: "swimming-pool",
    5: "vehicle",
    6: "person",
    7: "wind-mill",
}

CANONICAL_NAME_TO_ID: dict[str, int] = {
    name: cid for cid, name in CANONICAL_CLASSES.items()
}


@dataclass
class SahiConfig:
    """Configuration parameters for sliced inference and post-slice merging."""

    slice_height: int = 512
    slice_width: int = 512
    overlap_height_ratio: float = 0.20
    overlap_width_ratio: float = 0.20
    perform_standard_pred: bool = True
    postprocess_type: str = "GREEDYNMM"
    postprocess_match_metric: str = "IOU"
    postprocess_match_threshold: float = 0.60
    postprocess_class_agnostic: bool = False
    confidence_threshold: float = 0.001
    max_detections_per_image: int = 1500

    def __post_init__(self) -> None:
        """Validate parameter boundaries."""
        if self.slice_height <= 0 or self.slice_width <= 0:
            raise ValueError("Slice dimensions must be positive integers")
        if not (0.0 <= self.overlap_height_ratio < 1.0):
            raise ValueError("overlap_height_ratio must be in [0.0, 1.0)")
        if not (0.0 <= self.overlap_width_ratio < 1.0):
            raise ValueError("overlap_width_ratio must be in [0.0, 1.0)")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in [0.0, 1.0]")
        if not (0.0 <= self.postprocess_match_threshold <= 1.0):
            raise ValueError("postprocess_match_threshold must be in [0.0, 1.0]")
        if self.max_detections_per_image <= 0:
            raise ValueError("max_detections_per_image must be positive")
        valid_postprocess = {"NMS", "NMM", "GREEDYNMM"}
        if self.postprocess_type.upper() not in valid_postprocess:
            raise ValueError(f"postprocess_type must be one of {valid_postprocess}")
        valid_metrics = {"IOU", "IOS"}
        if self.postprocess_match_metric.upper() not in valid_metrics:
            raise ValueError(f"postprocess_match_metric must be one of {valid_metrics}")


def build_category_mapping(
    model_names: dict[int, str] | list[str],
    target_categories: list[dict[str, Any]] | dict[int, str] | None = None,
    smoke: bool = False,
) -> dict[int, int]:
    """Map model prediction indices to target dataset category IDs.

    Ensures exact name-based matching, never index assumptions.
    """
    names = dict(enumerate(model_names)) if isinstance(model_names, list) else model_names

    if target_categories is None:
        target = CANONICAL_NAME_TO_ID
    elif isinstance(target_categories, list):
        target = {cat["name"]: cat["id"] for cat in target_categories}
    else:
        target = {name: cid for cid, name in target_categories.items()}

    if (
        len(target) == 0
        or len(set(target.values())) != len(target)
        or len(set(names.values())) != len(names)
        or any(not isinstance(idx, int) or idx < 0 for idx in names)
    ):
        raise ValueError("Category names and IDs must be non-empty, unique, and non-negative")

    if not smoke and set(names.values()) != set(target):
        raise ValueError(
            "Model vocabulary differs from target ground truth; use smoke=True for partial tests"
        )

    mapping = {idx: target[name] for idx, name in names.items() if name in target}
    if not mapping:
        raise ValueError("No matching category names found between model and target")
    return mapping


class SahiInferenceRunner:
    """Executes Slicing-Aided Hyper Inference (SAHI) with unclipped coordinate re-projection."""

    def __init__(
        self,
        weights: str | Path | None = None,
        model_type: str = "ultralytics",
        device: str = "cpu",
        config: SahiConfig | None = None,
        detection_model: Any = None,
        category_map: dict[int, int] | None = None,
        smoke: bool = False,
    ) -> None:
        self.config = config or SahiConfig()
        self.device = device
        self.smoke = smoke

        if detection_model is not None:
            self.detection_model = detection_model
        else:
            if weights is None:
                raise ValueError("Must provide either weights path or pre-initialized detection_model")
            weights_path = Path(weights)
            if not weights_path.is_file():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type=model_type,
                model_path=str(weights_path),
                confidence_threshold=self.config.confidence_threshold,
                device=device,
                image_size=self.config.slice_height,
            )

        # Resolve category mapping
        raw_names = getattr(self.detection_model, "category_names", None)
        if raw_names is None and hasattr(self.detection_model, "model"):
            raw_names = getattr(self.detection_model.model, "names", None)
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
        """Run sliced inference on a single image and return standard COCO prediction records.

        Bounding boxes are unclipped [x_min, y_min, width, height] in original pixel coordinates.
        """
        start_time = time.perf_counter()

        sahi_result = get_sliced_prediction(
            image=image if isinstance(image, (np.ndarray, Image.Image)) else str(image),
            detection_model=self.detection_model,
            slice_height=self.config.slice_height,
            slice_width=self.config.slice_width,
            overlap_height_ratio=self.config.overlap_height_ratio,
            overlap_width_ratio=self.config.overlap_width_ratio,
            perform_standard_pred=self.config.perform_standard_pred,
            postprocess_type=self.config.postprocess_type.upper(),
            postprocess_match_metric=self.config.postprocess_match_metric.upper(),
            postprocess_match_threshold=self.config.postprocess_match_threshold,
            postprocess_class_agnostic=self.config.postprocess_class_agnostic,
            force_postprocess_type=True,
            verbose=0,
        )

        total_duration_ms = (time.perf_counter() - start_time) * 1000.0

        durations = {
            "slice_ms": sahi_result.durations_in_seconds.get("slice", 0.0) * 1000.0,
            "prediction_ms": sahi_result.durations_in_seconds.get("prediction", 0.0) * 1000.0,
            "postprocess_ms": sahi_result.durations_in_seconds.get("postprocess", 0.0) * 1000.0,
            "total_ms": total_duration_ms,
        }

        detections: list[dict[str, Any]] = []
        for pred in sahi_result.object_prediction_list:
            model_cat_id = pred.category.id
            if model_cat_id not in self.category_map:
                continue

            target_cat_id = self.category_map[model_cat_id]
            score = float(pred.score.value)
            if not (0.0 <= score <= 1.0) or not math.isfinite(score):
                continue

            bbox_xywh = pred.bbox.to_xywh()
            x_min, y_min, width, height = (float(v) for v in bbox_xywh)

            if not all(math.isfinite(v) for v in [x_min, y_min, width, height]):
                continue
            if width <= 0 or height <= 0:
                continue

            detections.append(
                {
                    "image_id": image_id,
                    "category_id": target_cat_id,
                    "bbox": [x_min, y_min, width, height],
                    "score": score,
                }
            )

        # Sort descending by confidence score and cap detections
        detections.sort(key=lambda d: d["score"], reverse=True)
        if len(detections) > self.config.max_detections_per_image:
            detections = detections[: self.config.max_detections_per_image]

        return detections, durations


def export_sahi_predictions(
    weights: Path,
    data: Path,
    output_dir: Path,
    *,
    split: str = "val",
    config: SahiConfig | None = None,
    annotations: Path | None = None,
    image_root: Path | None = None,
    image_ids: list[int] | None = None,
    device: str = "cpu",
    smoke: bool = False,
) -> dict[str, Any]:
    """Execute SAHI sliced inference on a dataset split and export standard predictions JSON."""
    weights = Path(weights)
    data = Path(data)
    output_dir = Path(output_dir)
    config = config or SahiConfig()

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
    if not selected_ids:
        raise ValueError("No images selected for inference")

    runner = SahiInferenceRunner(
        weights=weights,
        device=device,
        config=config,
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

    # Write standard predictions JSON
    pred_path.write_text(json.dumps(all_predictions, indent=2), encoding="utf-8")

    mean_total_ms = (
        float(np.mean([t["total_ms"] for t in timing_records])) if timing_records else 0.0
    )
    mean_slice_ms = (
        float(np.mean([t["slice_ms"] for t in timing_records])) if timing_records else 0.0
    )
    mean_pred_ms = (
        float(np.mean([t["prediction_ms"] for t in timing_records])) if timing_records else 0.0
    )
    mean_post_ms = (
        float(np.mean([t["postprocess_ms"] for t in timing_records])) if timing_records else 0.0
    )

    manifest = {
        "status": "complete",
        "strategy": "sahi_sliced",
        "weights": {"path": weights.as_posix(), "sha256": sha256(weights)},
        "annotations": {"path": resolved_ann.as_posix(), "sha256": sha256(resolved_ann)},
        "image_root": resolved_root.as_posix(),
        "sahi_config": asdict(config),
        "total_images": len(selected_ids),
        "total_detections": len(all_predictions),
        "zero_detection_image_ids": [iid for iid in selected_ids if detection_counts[str(iid)] == 0],
        "latency": {
            "mean_slice_ms": round(mean_slice_ms, 2),
            "mean_prediction_ms": round(mean_pred_ms, 2),
            "mean_postprocess_ms": round(mean_post_ms, 2),
            "mean_total_ms": round(mean_total_ms, 2),
            "throughput_fps": round(1000.0 / mean_total_ms, 2) if mean_total_ms > 0 else 0.0,
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
