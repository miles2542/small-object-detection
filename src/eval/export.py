"""Stream native Ultralytics detections to original-ID COCO results and a manifest."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from pathlib import Path

from src.eval.audit import sha256
from src.eval.protocol import evaluation_protocol


def _get_git_revision() -> str:
    """Safely obtain the current git commit SHA, falling back to 'unknown' if outside git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def category_mapping(model_names: dict | list, categories: list[dict], smoke: bool = False) -> dict:
    """Match exact class names, never numeric indices or semantic guesses."""
    names = dict(enumerate(model_names)) if isinstance(model_names, list) else model_names
    target = {category["name"]: category["id"] for category in categories}
    if (
        len(target) != len(categories)
        or len(set(target.values())) != len(categories)
        or len(set(names.values())) != len(names)
        or any(type(index) is not int or index < 0 for index in names)
    ):
        raise ValueError("Class names/IDs must be unique and model indices nonnegative integers")
    if not smoke and set(names.values()) != set(target):
        raise ValueError(
            "Checkpoint vocabulary differs from ground truth; use smoke=True only for smoke tests"
        )
    mapping = {index: target[name] for index, name in names.items() if name in target}
    if not mapping:
        raise ValueError("No exact shared class names")
    return mapping


def results_to_coco(result, image: dict, mapping: dict) -> list[dict]:
    """Results.xyxy is already rescaled to original pixels; do not undo letterboxing twice."""
    if tuple(result.orig_shape) != (image["height"], image["width"]):
        raise ValueError("Result dimensions differ from COCO image dimensions")
    if result.boxes is None:
        raise ValueError("Expected detection boxes, not segmentation/classification-only output")
    rows = []
    for box, score, category in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
        strict=True,
    ):
        if not all(math.isfinite(v) for v in box + [score, category]):
            raise ValueError("Non-finite prediction")
        if category != int(category) or int(category) not in mapping:
            raise ValueError("Invalid prediction class")

        x1 = max(0.0, min(float(image["width"]), float(box[0])))
        y1 = max(0.0, min(float(image["height"]), float(box[1])))
        x2 = max(0.0, min(float(image["width"]), float(box[2])))
        y2 = max(0.0, min(float(image["height"]), float(box[3])))
        score = max(0.0, min(1.0, float(score)))

        if x2 <= x1 or y2 <= y1:
            continue

        rows.append(
            {
                "image_id": image["id"],
                "category_id": mapping[int(category)],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score,
            }
        )
    return rows


def export_predictions(
    weights: Path,
    annotations: Path,
    image_root: Path,
    output_dir: Path,
    image_ids: list[int] | None = None,
    *,
    smoke: bool = False,
    device: str = "cpu",
    batch_size: int = 4,
) -> dict:
    """Export a complete image selection; partial-name pretrained runs are marked smoke-only.

    Use audited original or normalized COCO JSON. Outputs become complete
    only when manifest.json exists; failed runs remove their partial predictions.
    """
    import torch
    from ultralytics import YOLO

    weights, annotations = Path(weights), Path(annotations)
    image_root, output_dir = Path(image_root).resolve(), Path(output_dir)
    if not weights.is_file() or weights.suffix != ".pt":
        raise ValueError("Provide an existing native .pt checkpoint")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if any(
        (output_dir / name).exists()
        for name in ["predictions.json", "manifest.json", "predictions.json.part"]
    ):
        raise FileExistsError("Use a fresh output directory; previous exports are not overwritten")
    coco = json.loads(annotations.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in coco["images"]}
    if len(images) != len(coco["images"]):
        raise ValueError("Duplicate COCO image IDs")
    selected = list(images) if image_ids is None else list(image_ids)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= images.keys():
        raise ValueError("Select a nonempty set of distinct, known image IDs")
    paths = {}
    for iid in selected:
        path = (image_root / images[iid]["file_name"]).resolve()
        if not path.is_relative_to(image_root) or not path.is_file() or path in paths:
            raise ValueError(f"Missing, duplicate or out-of-root image path: {path}")
        paths[path] = images[iid]
    model = YOLO(str(weights), task="detect")
    if model.task != "detect":
        raise ValueError("Only bounding-box detection models are supported")
    mapping = category_mapping(model.names, coco["categories"], smoke)
    protocol = evaluation_protocol()
    native_end2end = getattr(model.model.model[-1], "one2one_cv2", None) is not None
    settings = {
        "imgsz": protocol["image_size"],
        "conf": protocol["export_confidence"],
        "iou": protocol["nms_iou"],
        "max_det": protocol["export_max_detections"],
        "agnostic_nms": protocol["class_agnostic_nms"],
        "augment": False,
        "rect": False,
        "device": device,
        "batch": batch_size,
        "classes": sorted(mapping) if smoke else None,
        "verbose": False,
        "save": False,
        "seed": 42,
        "nms": False if native_end2end else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "predictions.json.part"
    counts, seen, head_limits = {}, set(), set()
    start, total = time.perf_counter(), 0
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write("[")
            sources = list(paths)
            for offset in range(0, len(sources), batch_size):
                batch = sources[offset : offset + batch_size]
                pending = set(batch)
                for result in model.predict(
                    source=[str(p) for p in batch], stream=True, **settings
                ):
                    path = Path(result.path).resolve()
                    if path not in pending:
                        raise ValueError("Unexpected or repeated result path")
                    pending.remove(path)
                    image = paths[path]
                    rows = results_to_coco(result, image, mapping)
                    if len(rows) > settings["max_det"]:
                        raise ValueError("Prediction count exceeds requested cap")
                    if rows:
                        stream.write(
                            ("," if total else "") + json.dumps(rows, allow_nan=False)[1:-1]
                        )
                    total += len(rows)
                    seen.add(image["id"])
                    counts[str(image["id"])] = len(rows)
                    head = model.predictor.model.model.model[-1]
                    if native_end2end and not getattr(head, "end2end", False):
                        raise ValueError("Requested native end-to-end head was not enabled")
                    if getattr(head, "end2end", False):
                        head_limits.add(int(head.max_det))
                        if head.max_det < settings["max_det"]:
                            raise ValueError(
                                "End-to-end head truncates below requested detection cap"
                            )
                if pending:
                    raise ValueError("Predictor omitted selected images")
            stream.write("]")
        if seen != set(selected):
            raise ValueError("Export did not cover the full image selection")
        elapsed = time.perf_counter() - start
        manifest = {
            "status": "complete",
            "purpose": "smoke_only" if smoke else "evaluation",
            "weights": {"path": weights.as_posix(), "sha256": sha256(weights)},
            "annotations": {"path": annotations.as_posix(), "sha256": sha256(annotations)},
            "image_root": image_root.as_posix(),
            "image_ids": selected,
            "images": [{"id": iid, "file_name": images[iid]["file_name"]} for iid in selected],
            "category_mapping": {
                str(i): {"name": model.names[i], "category_id": cid} for i, cid in mapping.items()
            },
            "excluded_model_classes": [name for i, name in model.names.items() if i not in mapping],
            "prediction_counts": counts,
            "detections": total,
            "zero_detection_image_ids": [iid for iid in selected if not counts[str(iid)]],
            "cap_hit_image_ids": [
                iid for iid in selected if counts[str(iid)] == settings["max_det"]
            ],
            "head_max_dets": sorted(head_limits),
            "postprocessing": "native_end_to_end" if head_limits else "class_aware_nms",
            "protocol": protocol,
            "predict_settings": settings,
            "elapsed_seconds": elapsed,
            "effective_device": str(model.predictor.device),
            "torch_num_threads": torch.get_num_threads(),
            "torch_build": torch.__version__,
            "python": platform.python_version(),
            "repository_revision": _get_git_revision(),
            "export_code_sha256": sha256(Path(__file__)),
            "predictions_sha256": sha256(temporary),
        }
        temporary.replace(output_dir / "predictions.json")
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    finally:
        temporary.unlink(missing_ok=True)
