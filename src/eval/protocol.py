"""Explicit evaluation settings and a non-mutating COCO ground-truth adapter."""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

REFERENCE_REVISION = "44a230ae5197cb89bf9e5e62f313cac3ad30c7af"
REFERENCE_SHA256 = "2c3f84612f448dccf085c72f20711f5fb8dcac3022e8f24cdf04af283a24fa1e"
REFERENCE_URL = (
    f"https://github.com/jwwangchn/cocoapi-aitod/blob/{REFERENCE_REVISION}/"
    "aitodpycocotools/aitodpycocotools/cocoeval.py"
)


def evaluation_protocol(name: str = "standard") -> dict:
    """Return fresh, JSON-compatible settings; area endpoints follow COCOeval."""
    if name != "standard":
        raise ValueError(f"Unknown protocol {name!r}; expected 'standard'")
    return {
        "name": "aitod-standard-v1",
        "area_labels": ["all", "verytiny", "tiny", "small", "medium"],
        "area_ranges": [[0, 1e10], [0, 64], [64, 256], [256, 1024], [1024, 1e10]],
        "iou_thresholds": np.linspace(0.5, 0.95, 10).tolist(),
        "recall_thresholds": np.linspace(0.0, 1.0, 101).tolist(),
        "max_dets": [1, 100, 1500],
        "use_categories": True,
        "export_confidence": 0.001,
        "export_max_detections": 1500,
        "operating_confidence": 0.25,
        "nms_iou": 0.7,
        "image_size": 800,
        "augment": False,
        "class_agnostic_nms": False,
        "ground_truth_policy": {
            "invalid_boxes": "drop_nonpositive_dimensions_or_area",
            "annotation_ids": "positive_ids_with_source_annotation_id",
            "coordinates": "original_unclipped",
            "area": "original_annotation_area",
        },
        "reference_revision": REFERENCE_REVISION,
    }


def area_memberships(area: float) -> list[str]:
    """Closed intervals deliberately allow shared boundary membership."""
    protocol = evaluation_protocol()
    return [
        label
        for label, (low, high) in zip(protocol["area_labels"], protocol["area_ranges"])
        if math.isfinite(area) and low <= area <= high
    ]


def prepare_ground_truth(coco: dict, image_ids: list[int] | None = None) -> dict:
    """Select original images, drop invalid boxes and replace zero-based annotation IDs.

    Image/category IDs, positive box geometry and annotated area stay unchanged.
    Source annotation IDs remain available for tracing qualitative examples.
    """
    result = deepcopy(coco)
    selected = {image["id"] for image in result["images"]}
    if image_ids is not None:
        requested = set(image_ids)
        if len(requested) != len(image_ids) or not requested <= selected:
            raise ValueError("Image selection contains duplicate or unknown IDs")
        selected = requested
    result["images"] = [image for image in result["images"] if image["id"] in selected]
    annotations = []
    for ann in result["annotations"]:
        if ann["image_id"] not in selected:
            continue
        box = ann["bbox"]
        if len(box) != 4 or not all(math.isfinite(v) for v in box + [ann["area"]]):
            raise ValueError(f"Malformed/non-finite annotation {ann['id']}")
        if box[2] <= 0 or box[3] <= 0 or ann["area"] <= 0:
            continue
        ann["source_annotation_id"] = ann.get("source_annotation_id", ann["id"])
        ann["id"] = len(annotations) + 1
        ann.setdefault("iscrowd", 0)
        annotations.append(ann)
    result["annotations"] = annotations
    result.setdefault("info", {})
    return result
