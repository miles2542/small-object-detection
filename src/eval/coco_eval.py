"""COCO bbox matching with explicit AI-TOD ranges and confidence-point summaries."""

from __future__ import annotations

import io
import json
import math
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.eval.audit import sha256
from src.eval.protocol import evaluation_protocol, prepare_ground_truth


def valid_mean(values) -> float | None:
    """Exclude COCO's unsupported -1 entries; serialize absent metrics as JSON null."""
    values = np.asarray(values)
    eligible = values[values >= 0]
    return float(eligible.mean()) if eligible.size else None


def validate_inputs(coco: dict, predictions: list, selected: list[int] | None) -> None:
    if not isinstance(coco, dict) or not {"images", "categories", "annotations"} <= coco.keys():
        raise ValueError("Ground truth must be COCO JSON with images, categories and annotations")
    images = {image["id"] for image in coco["images"]}
    categories = {category["id"] for category in coco["categories"]}
    if any(type(value) is not int for value in images | categories):
        raise ValueError("COCO image/category IDs must be integers")
    if (
        not images
        or not categories
        or len(images) != len(coco["images"])
        or len(categories) != len(coco["categories"])
        or len({a["id"] for a in coco["annotations"]}) != len(coco["annotations"])
    ):
        raise ValueError(
            "Ground truth needs nonempty unique image/category IDs and unique annotation IDs"
        )
    for ann in coco["annotations"]:
        if ann["image_id"] not in images or ann["category_id"] not in categories:
            raise ValueError("Annotation references an unknown image/category")
        if ann.get("ignore", 0) and not ann.get("iscrowd", 0):
            raise ValueError(
                "Standalone ignore=1 is ambiguous in COCO bbox; use an explicit dataset policy"
            )
    if selected is not None:
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= images:
            raise ValueError("Evaluation needs distinct known image IDs and a nonempty selection")
        images = set(selected)
    if not isinstance(predictions, list):
        raise TypeError("Predictions must be a COCO results array")
    for row in predictions:
        if (
            not isinstance(row, dict)
            or not {"image_id", "category_id", "bbox", "score"} <= row.keys()
        ):
            raise ValueError("Malformed COCO result row")
        box, score = row["bbox"], row["score"]
        if (
            type(row["image_id"]) is not int
            or type(row["category_id"]) is not int
            or row["image_id"] not in images
            or row["category_id"] not in categories
            or len(box) != 4
            or not all(math.isfinite(v) for v in box + [score])
            or box[2] <= 0
            or box[3] <= 0
            or not 0 <= score <= 1
        ):
            raise ValueError("Unknown ID, invalid geometry or non-finite/out-of-range confidence")


def operating_counts(evaluator: COCOeval, area_index: int, threshold: float) -> dict:
    """Score-filter existing matches; lower-score detections cannot displace higher scores."""
    p = evaluator.params
    counts = np.zeros((len(p.iouThrs), len(p.catIds), 3), dtype=np.int64)
    image_count, area_count = len(p.imgIds), len(p.areaRng)
    for k in range(len(p.catIds)):
        start = (k * area_count + area_index) * image_count
        for entry in evaluator.evalImgs[start : start + image_count]:
            if entry is None:
                continue
            support = int(np.count_nonzero(~np.asarray(entry["gtIgnore"], dtype=bool)))
            keep = np.asarray(entry["dtScores"]) >= threshold
            matched = entry["dtMatches"][:, keep] > 0
            eligible = ~entry["dtIgnore"][:, keep]
            tp = (matched & eligible).sum(axis=1)
            fp = (~matched & eligible).sum(axis=1)
            counts[:, k, 0] += tp
            counts[:, k, 1] += fp
            counts[:, k, 2] += support - tp

    def summarize(values: np.ndarray) -> dict:
        tp, fp, fn = values.T
        precision = np.divide(tp, tp + fp, out=np.full(len(tp), np.nan), where=tp + fp > 0)
        recall = np.divide(tp, tp + fn, out=np.full(len(tp), np.nan), where=tp + fn > 0)
        return {
            "tp": tp.tolist(),
            "fp": fp.tolist(),
            "fn": fn.tolist(),
            "precision": [None if np.isnan(v) else float(v) for v in precision],
            "recall": [None if np.isnan(v) else float(v) for v in recall],
        }

    per_class = {str(cid): summarize(counts[:, k]) for k, cid in enumerate(p.catIds)}
    for k, cid in enumerate(p.catIds):
        for t in range(len(p.iouThrs)):
            if counts[t, k, 0] + counts[t, k, 2] == 0:
                per_class[str(cid)]["precision"][t] = None
            elif per_class[str(cid)]["precision"][t] is None:
                per_class[str(cid)]["precision"][t] = 0.0
    micro = summarize(counts.sum(axis=1))
    macro_p, macro_r = [], []
    for t in range(len(p.iouThrs)):
        supported = counts[t, :, 0] + counts[t, :, 2] > 0
        macro_p.append(
            float(
                np.mean(
                    [
                        per_class[str(cid)]["precision"][t] or 0
                        for k, cid in enumerate(p.catIds)
                        if supported[k]
                    ]
                )
            )
            if supported.any()
            else None
        )
        macro_r.append(
            float(
                np.mean(
                    [
                        per_class[str(cid)]["recall"][t]
                        for k, cid in enumerate(p.catIds)
                        if supported[k]
                    ]
                )
            )
            if supported.any()
            else None
        )
    return {
        "score_threshold": threshold,
        "iou_thresholds": p.iouThrs.tolist(),
        "micro": micro,
        "macro": {
            "precision": macro_p,
            "recall": macro_r,
            "mean_precision": valid_mean([v for v in macro_p if v is not None]),
            "mean_recall": valid_mean([v for v in macro_r if v is not None]),
        },
        "per_class": per_class,
        "micro_mean_precision": valid_mean([v for v in micro["precision"] if v is not None]),
        "micro_mean_recall": valid_mean([v for v in micro["recall"] if v is not None]),
    }


def evaluate_coco(
    coco: dict,
    predictions: list,
    *,
    image_ids: list[int] | None = None,
    protocol_name: str = "assignment",
    confidence: float | None = None,
    max_detections: int | None = None,
) -> tuple[dict, COCOeval]:
    """Evaluate all selected images, including negatives and images without predictions.

    COCO crowd and area-ignore behavior is retained. Standalone non-crowd ignore
    flags are rejected rather than silently overwritten by COCOeval._prepare().
    All returned scores are fractions; unsupported values are None.
    """
    validate_inputs(coco, predictions, image_ids)
    protocol = evaluation_protocol(protocol_name)
    if max_detections is not None:
        if type(max_detections) is not int or max_detections < 1:
            raise ValueError("max_detections must be a positive integer")
        protocol["max_dets"] = sorted({1, min(100, max_detections), max_detections})
    confidence = protocol["operating_confidence"] if confidence is None else confidence
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Operating confidence must be in [0,1]")
    protocol["operating_confidence"] = confidence
    ground_truth = prepare_ground_truth(coco, image_ids)
    with redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = ground_truth
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(deepcopy(predictions))
        else:
            dt = COCO()
            dt.dataset = {
                "images": deepcopy(ground_truth["images"]),
                "categories": deepcopy(ground_truth["categories"]),
                "annotations": [],
            }
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        p = evaluator.params
        p.areaRng, p.areaRngLbl = protocol["area_ranges"], protocol["area_labels"]
        p.maxDets = protocol["max_dets"]
        p.iouThrs, p.recThrs = (
            np.array(protocol["iou_thresholds"]),
            np.array(protocol["recall_thresholds"]),
        )
        evaluator.evaluate()
        evaluator.accumulate()
    precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]
    i50, i75 = [int(np.flatnonzero(np.isclose(p.iouThrs, value))[0]) for value in [0.5, 0.75]]
    summary = {
        "units": "fraction",
        "protocol": protocol,
        "image_ids": list(map(int, p.imgIds)),
        "category_ids": list(map(int, p.catIds)),
        "bins": {},
        "per_class": {},
    }
    for area, label in enumerate(p.areaRngLbl):
        point = operating_counts(evaluator, area, confidence)
        summary["bins"][label] = {
            "AP": valid_mean(precision[:, :, :, area, -1]),
            "AP50": valid_mean(precision[i50, :, :, area, -1]),
            "AP75": valid_mean(precision[i75, :, :, area, -1]),
            "AR": valid_mean(recall[:, :, area, -1]),
            "AR_by_max_dets": {
                str(cap): valid_mean(recall[:, :, area, m]) for m, cap in enumerate(p.maxDets)
            },
            "support": point["micro"]["tp"][0] + point["micro"]["fn"][0],
            "operating_point": point,
        }
    for k, cid in enumerate(p.catIds):
        summary["per_class"][str(cid)] = {
            "name": gt.cats[cid]["name"],
            "bins": {
                label: {
                    "AP": valid_mean(precision[:, :, k, a, -1]),
                    "AP50": valid_mean(precision[i50, :, k, a, -1]),
                    "AR": valid_mean(recall[:, k, a, -1]),
                    "support": sum(
                        summary["bins"][label]["operating_point"]["per_class"][str(cid)][key][0]
                        for key in ["tp", "fn"]
                    ),
                }
                for a, label in enumerate(p.areaRngLbl)
            },
        }
    return summary, evaluator


def evaluate_files(
    annotations: Path,
    predictions: Path,
    output_dir: Path,
    *,
    protocol_name: str = "assignment",
    allow_smoke: bool = False,
    confidence: float | None = None,
) -> dict:
    """Verify an optional sibling export manifest, evaluate, and save JSON plus PR tensors."""
    annotations, predictions, output_dir = Path(annotations), Path(predictions), Path(output_dir)
    if any((output_dir / name).exists() for name in ["metrics.json", "curves.npz"]):
        raise FileExistsError("Use a fresh evaluation output directory")
    coco, rows = (
        json.loads(annotations.read_text(encoding="utf-8")),
        json.loads(predictions.read_text(encoding="utf-8")),
    )
    manifest_path = predictions.with_name("manifest.json")
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    )
    selected = None
    if manifest_path.exists():
        if (
            not isinstance(manifest, dict)
            or manifest.get("status") != "complete"
            or sha256(predictions) != manifest.get("predictions_sha256")
        ):
            raise ValueError("Incomplete export or prediction fingerprint mismatch")
        if sha256(annotations) != manifest["annotations"]["sha256"]:
            raise ValueError("Ground-truth fingerprint differs from the prediction export")
        if manifest["purpose"] == "smoke_only" and not allow_smoke:
            raise ValueError(
                "Smoke predictions require allow_smoke=True; they are not benchmark results"
            )
        operating = (
            evaluation_protocol(protocol_name)["operating_confidence"]
            if confidence is None
            else confidence
        )
        floor = manifest.get("predict_settings", {}).get("conf", 0)
        if operating < floor:
            raise ValueError("Operating confidence is below the recorded prediction export floor")
        selected = manifest["image_ids"]
        counts = Counter(row["image_id"] for row in rows)
        if len(rows) != manifest["detections"] or any(
            counts[iid] != manifest["prediction_counts"][str(iid)] for iid in selected
        ):
            raise ValueError("Export manifest counts disagree with predictions")
    metrics, evaluator = evaluate_coco(
        coco, rows, image_ids=selected, protocol_name=protocol_name, confidence=confidence
    )
    metrics["provenance"] = {
        "purpose": manifest["purpose"] if manifest else "external_predictions_unverified",
        "annotations_sha256": sha256(annotations),
        "predictions_sha256": sha256(predictions),
        "export_manifest": manifest,
        "pycocotools_version": version("pycocotools"),
        "evaluator_code_sha256": sha256(Path(__file__)),
        "protocol_code_sha256": sha256(Path(__file__).with_name("protocol.py")),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "curves.npz",
        precision=evaluator.eval["precision"],
        recall=evaluator.eval["recall"],
        scores=evaluator.eval["scores"],
        iou_thresholds=evaluator.params.iouThrs,
        recall_thresholds=evaluator.params.recThrs,
        category_ids=evaluator.params.catIds,
        area_labels=evaluator.params.areaRngLbl,
        max_dets=evaluator.params.maxDets,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metrics
