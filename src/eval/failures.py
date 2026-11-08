"""Reproducible TIDE diagnostics for saved COCO predictions."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tidecv.data import Data
    from tidecv.quantify import TIDERun

from src.eval.audit import sha256
from src.eval.coco_eval import evaluate_coco, validate_inputs
from src.eval.protocol import evaluation_protocol, prepare_ground_truth

ERRORS = ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss")
SIZE_BINS = ("verytiny", "tiny", "small", "medium")


def exclusive_size_bin(area: float) -> str:
    """Assign one event to one bin; official AP ranges remain closed at boundaries."""
    if area < 64:
        return "verytiny"
    if area < 256:
        return "tiny"
    if area < 1024:
        return "small"
    return "medium"


def _tide_data(
    coco: dict[str, Any], rows: list[dict[str, Any]], max_dets: int
) -> tuple[Data, Data]:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sod-matplotlib"))
    from tidecv.data import Data

    gt = Data("ground_truth", max_dets=max_dets)
    pred = Data("predictions", max_dets=max_dets)
    for category in coco["categories"]:
        gt.add_class(category["id"], category["name"])
        pred.add_class(category["id"], category["name"])
    for image in coco["images"]:
        gt.add_image(image["id"], image["file_name"])
        pred.add_image(image["id"], image["file_name"])
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            gt.add_ignore_region(ann["image_id"], ann["category_id"], ann["bbox"])
        else:
            gt.add_ground_truth(ann["image_id"], ann["category_id"], ann["bbox"])
        gt.annotations[-1]["source_annotation_id"] = ann.get("source_annotation_id", ann["id"])
        gt.annotations[-1]["area"] = ann["area"]
    for index, row in enumerate(rows):
        pred.add_detection(row["image_id"], row["category_id"], row["score"], row["bbox"])
        pred.annotations[-1]["source_prediction_index"] = index
    return gt, pred


def _error_events(run: TIDERun, gt_data: Data) -> list[dict[str, Any]]:
    from tidecv.errors.main_errors import (
        BackgroundError,
        BoxError,
        ClassError,
        DuplicateError,
        MissedError,
        OtherError,
    )

    names = {
        ClassError: "Cls",
        BoxError: "Loc",
        OtherError: "Both",
        DuplicateError: "Dupe",
        BackgroundError: "Bkg",
        MissedError: "Miss",
    }
    events = []
    for error in run.errors:
        kind = names[type(error)]
        prediction = getattr(error, "pred", None)
        truth = getattr(error, "gt", None)
        if truth is None and kind == "Dupe":
            matched_id = error.suppressor.get("matched_with")
            if matched_id is not None:
                truth = gt_data.annotations[matched_id]
        anchor = truth if truth is not None else prediction
        area = anchor.get("area") if truth is not None else anchor["bbox"][2] * anchor["bbox"][3]
        item = {
            "type": kind,
            "size_bin": exclusive_size_bin(area),
            "image_id": int(anchor["image"]),
            "category_id": int(anchor["class"]),
            "area_px2": float(area),
            "gt_bbox": None if truth is None else list(truth["bbox"]),
            "prediction_bbox": None if prediction is None else list(prediction["bbox"]),
            "score": None if prediction is None else float(prediction["score"]),
            "source_annotation_id": None if truth is None else truth["source_annotation_id"],
            "source_prediction_index": (
                None if prediction is None else prediction["source_prediction_index"]
            ),
        }
        if kind == "Loc" and truth is not None and prediction is not None:
            tx, ty, tw, th = truth["bbox"]
            px, py, pw, ph = prediction["bbox"]
            dx = px + pw / 2 - tx - tw / 2
            dy = py + ph / 2 - ty - th / 2
            item["center_shift_px"] = [dx, dy]
            item["center_distance_px"] = math.hypot(dx, dy)
            item["center_distance_over_sqrt_gt_area"] = (
                math.hypot(dx, dy) / math.sqrt(area) if area > 0 else 0.0
            )
        events.append(item)
    return events


def _check_manifest(path: Path, gt_hash: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    manifest_path = path.with_name("manifest.json")
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("predictions_sha256") != sha256(path)
        or manifest.get("annotations", {}).get("sha256") != gt_hash
        or manifest.get("detections") != len(rows)
    ):
        raise ValueError(f"Invalid export manifest beside {path}")
    ids = manifest.get("image_ids")
    if not isinstance(ids, list) or len(ids) != len(set(ids)):
        raise ValueError(f"Invalid image IDs in export manifest beside {path}")
    counts = Counter(row["image_id"] for row in rows)
    if not set(counts) <= set(ids):
        raise ValueError(f"Predictions include images outside export manifest beside {path}")
    if any(
        counts[image_id] != manifest.get("prediction_counts", {}).get(str(image_id), 0)
        for image_id in ids
    ):
        raise ValueError(f"Prediction counts disagree with export manifest beside {path}")
    return manifest


def _render_charts(report: dict[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = list(report["models"])
    positions = np.arange(len(ERRORS))
    width = 0.8 / len(models)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, name in enumerate(models):
        data = report["models"][name]
        x = positions + (i - (len(models) - 1) / 2) * width
        axes[0].bar(
            x,
            [np.nan if data["delta_ap_pp"][e] is None else data["delta_ap_pp"][e] for e in ERRORS],
            width,
            label=name,
        )
        axes[1].bar(x, [data["error_counts"][e] for e in ERRORS], width, label=name)
    for ax, ylabel in zip(axes, ("Oracle ΔAP50 (percentage points)", "Error events (count)")):
        ax.set_xticks(positions, ERRORS)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    fig.suptitle("TIDE failure modes for saved predictions")
    fig.tight_layout()
    fig.savefig(output_dir / "tide_breakdown.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(models), 1, figsize=(10, 3.5 * len(models)), squeeze=False)
    colors = plt.get_cmap("tab10").colors
    for ax, name in zip(axes[:, 0], models):
        data = report["models"][name]
        bottom = np.zeros(len(SIZE_BINS))
        for i, kind in enumerate(ERRORS):
            values = np.array([data["size_bins"][b]["percent"][kind] for b in SIZE_BINS])
            ax.bar(SIZE_BINS, values, bottom=bottom, label=kind, color=colors[i])
            bottom += values
        ax.set_ylim(0, 100)
        ax.set_ylabel("Share of error events (%)")
        ax.set_title(name)
        ax.legend(ncol=6, fontsize=8)
        if not any(data["size_bins"][size]["total_error_events"] for size in SIZE_BINS):
            ax.text(
                0.5,
                0.5,
                "No TIDE error events",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="0.4",
            )
    fig.suptitle("Failure composition by exclusive event-size bin")
    fig.tight_layout()
    fig.savefig(output_dir / "failure_by_size.png", dpi=180)
    plt.close(fig)


def _render_gallery(
    events_by_model: dict[str, list[dict[str, Any]]],
    coco: dict[str, Any],
    image_root: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    images = {image["id"]: image for image in coco["images"]}
    selected = []
    # Deterministic examples for the main micro-object diagnostic modes.
    for name, events in events_by_model.items():
        for size in SIZE_BINS:
            for kind in ("Miss", "Loc", "Bkg"):
                candidates = [e for e in events if e["size_bin"] == size and e["type"] == kind]
                if candidates:
                    chosen = min(
                        candidates,
                        key=lambda e: (
                            e["image_id"],
                            str(e["source_annotation_id"]),
                            e["source_prediction_index"] or -1,
                        ),
                    )
                    selected.append({"model": name, **chosen})
    if not selected:
        return []
    fig, axes = plt.subplots(len(selected), 2, figsize=(9, 3.0 * len(selected)), squeeze=False)
    root = image_root.resolve()
    for pair, item in zip(axes, selected):
        path = (root / images[item["image_id"]]["file_name"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image path escapes --image-root: {path}")
        with Image.open(path) as source:
            pixels = source.convert("RGB")
            box = item["gt_bbox"] or item["prediction_bbox"]
            x, y, w, h = box
            pad = max(24, 3 * max(w, h))
            cx, cy = x + w / 2, y + h / 2
            left, right = max(0, cx - pad), min(pixels.width, cx + pad)
            top, bottom = max(0, cy - pad), min(pixels.height, cy + pad)
            for col, ax in enumerate(pair):
                ax.imshow(pixels)
                if item["gt_bbox"] is not None:
                    ax.add_patch(
                        Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=1.5)
                    )
                prediction_box = item["prediction_bbox"]
                if prediction_box is not None:
                    ax.add_patch(
                        Rectangle(
                            (prediction_box[0], prediction_box[1]),
                            prediction_box[2],
                            prediction_box[3],
                            fill=False,
                            edgecolor="red",
                            linewidth=1.5,
                        )
                    )
                if col:
                    ax.set_xlim(left, right)
                    ax.set_ylim(bottom, top)
                ax.set_title(
                    f"{item['model']} | {item['size_bin']} | {item['type']} | image {item['image_id']}"
                    + (" | crop" if col else " | full"),
                    fontsize=9,
                )
                ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "failure_gallery.png", dpi=150)
    plt.close(fig)
    return selected


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TIDE failure analysis",
        "",
        f"Ground truth: `{report['ground_truth']}` (SHA-256 `{report['ground_truth_sha256']}`).",
        "",
        (
            "TIDE uses IoU 0.50, background IoU 0.10 and ranked saved predictions. "
            "Oracle ΔAP is in percentage points; it is not an additive decomposition. "
            "The six event counts are mutually exclusive under TIDE's assignment."
        ),
        "",
        *[f"- Comparability: {warning}" for warning in report["comparability_warnings"]],
        "",
        "| Model | TIDE AP50 (%) | Cls | Loc | Both | Dupe | Bkg | Miss |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, data in report["models"].items():
        values = " | ".join(
            "N/A" if data["delta_ap_pp"][key] is None else f"{data['delta_ap_pp'][key]:.3f}"
            for key in ERRORS
        )
        lines.append(f"| {name} | {data['tide_ap50_percent']:.3f} | {values} |")
    lines += [
        "",
        (
            "Official COCO AP (%) by size, under the repository's closed area bins. "
            "TIDE AP50 may differ because its matcher and unsupported-class treatment differ:"
        ),
        "",
        "| Model | All | Very tiny | Tiny | Small | Medium |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, data in report["models"].items():
        values = " | ".join(
            "N/A"
            if data["official_ap_percent"][size] is None
            else f"{data['official_ap_percent'][size]:.3f}"
            for size in ("all", *SIZE_BINS)
        )
        lines.append(f"| {name} | {values} |")
    lines += ["", "## Error counts and size distribution", ""]
    for name, data in report["models"].items():
        lines += [
            f"### {name}",
            "",
            (
                f"Predictions: `{data['predictions']}` (SHA-256 `{data['predictions_sha256']}`). "
                f"Export manifest: {'verified' if data['verified_manifest'] else 'absent; provenance unverified'}."
            ),
            "",
            "| Event size | GT support | Cls | Loc | Both | Dupe | Bkg | Miss | Loc center shift median (px) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for size in SIZE_BINS:
            row = data["size_bins"][size]
            counts = " | ".join(str(row["counts"][kind]) for kind in ERRORS)
            shift = row["loc_center_shift_median_px"]
            lines.append(
                f"| {size} | {row['gt_support']} | {counts} | {shift:.3f} |"
                if shift is not None
                else f"| {size} | {row['gt_support']} | {counts} | N/A |"
            )
        lines.append("")
        for size in SIZE_BINS:
            row = data["size_bins"][size]
            if row["total_error_events"]:
                dominant = max(ERRORS, key=lambda kind: row["counts"][kind])
                lines.append(
                    f"- {size}: {dominant} is most frequent "
                    f"({row['percent'][dominant]:.1f}% of {row['total_error_events']} events)."
                )
        tiny = data["size_bins"]["verytiny"]
        medium = data["size_bins"]["medium"]
        if tiny["total_error_events"] and medium["total_error_events"]:
            lines.append(
                "- Very-tiny minus medium event share (percentage points): "
                + ", ".join(
                    f"{kind} {tiny['percent'][kind] - medium['percent'][kind]:+.1f}"
                    for kind in ("Miss", "Loc", "Cls", "Bkg")
                )
                + "."
            )
        lines.append("")
    lines += [
        (
            "The size chart shows each error type's percentage within a bin. Event bins are "
            "exclusive: [0,64), [64,256), [256,1024), [1024,∞) px². "
            "Official AP bins are closed at both endpoints, so their supports differ at boundaries. "
            "Cls/Loc/Miss and Dupe use linked GT area; Both/Bkg use prediction area."
        ),
        "",
        (
            "Miss counts exclude ground truths whose failure is already explained by another TIDE error. "
            "Bkg counts are assigned by predicted box area. Neither count alone is an official recall rate."
        ),
        "",
        (
            "Localization center shift is measured between predicted and GT box centers for TIDE Loc events. "
            "It is descriptive and does not prove a causal mechanism."
        ),
        "N/A oracle values indicate TIDE cannot define AP after that correction removes all GT support.",
        "",
    ]
    if report["gallery_examples"]:
        lines += [
            "## Qualitative examples",
            "",
            "See `failure_gallery.png` for original context and crop; green is GT, red is prediction.",
            "",
            "| Model | Size | Error | Image ID | Annotation ID | Prediction score | Observation |",
            "| :--- | :--- | :--- | ---: | ---: | ---: | :--- |",
        ]
        for item in report["gallery_examples"]:
            if item["type"] == "Loc":
                note = f"Box centers differ by {item['center_distance_px']:.2f} px."
            elif item["type"] == "Miss":
                note = "GT remains missed after TIDE error attribution."
            else:
                note = "Detection is attributed to background by TIDE."
            score = "N/A" if item["score"] is None else f"{item['score']:.3f}"
            annotation = (
                "N/A" if item["source_annotation_id"] is None else str(item["source_annotation_id"])
            )
            lines.append(
                f"| {item['model']} | {item['size_bin']} | {item['type']} | "
                f"{item['image_id']} | {annotation} | {score} | {note} |"
            )
        lines += [
            "",
            "These notes describe TIDE attribution and geometry; inspect the image before drawing a causal conclusion.",
            "",
        ]
    else:
        lines += ["Qualitative gallery was not rendered: no image root or eligible examples.", ""]
    return "\n".join(lines)


def analyze_failures(
    ground_truth: Path,
    models: dict[str, Path],
    output_dir: Path,
    *,
    image_root: Path | None = None,
) -> dict[str, Any]:
    """Run official TIDE error attribution on comparable saved COCO outputs."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sod-matplotlib"))
    ground_truth, output_dir = Path(ground_truth), Path(output_dir)
    if not models or any(not name or not name.strip() for name in models):
        raise ValueError("Provide at least one labeled prediction JSON")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use a fresh failure-analysis output directory")
    coco_raw = json.loads(ground_truth.read_text(encoding="utf-8"))
    validate_inputs(coco_raw, [], None)
    coco = prepare_ground_truth(coco_raw)
    gt_hash = sha256(ground_truth)
    protocol = evaluation_protocol()
    max_dets = protocol["export_max_detections"]
    report = {
        "ground_truth": str(ground_truth.resolve()),
        "ground_truth_sha256": gt_hash,
        "tide_version": "1.0.1",
        "protocol": {
            "foreground_iou": 0.5,
            "background_iou": 0.1,
            "max_detections_per_image": max_dets,
        },
        "models": {},
        "gallery_examples": [],
        "comparability_warnings": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    from tidecv.quantify import TIDE, TIDERun

    all_events = {}
    image_sets = []
    settings = []
    for name, path in models.items():
        path = Path(path)
        rows = json.loads(path.read_text(encoding="utf-8"))
        validate_inputs(coco_raw, rows, None)
        manifest = _check_manifest(path, gt_hash, rows)
        image_sets.append(None if manifest is None else set(manifest["image_ids"]))
        settings.append(None if manifest is None else manifest.get("predict_settings"))
        model_coco = prepare_ground_truth(coco_raw, manifest["image_ids"] if manifest else None)
        official, _ = evaluate_coco(
            coco_raw, rows, image_ids=manifest["image_ids"] if manifest else None
        )
        gt_data, pred_data = _tide_data(model_coco, rows, max_dets)
        run = TIDERun(gt_data, pred_data, 0.5, 0.1, "bbox", max_dets)
        events = _error_events(run, gt_data)
        all_events[name] = events
        delta = {}
        for error_type in TIDE._error_types:
            try:
                delta[error_type.short_name] = float(
                    run.fix_main_errors(error_types=[error_type])[error_type]
                )
            except ZeroDivisionError:
                # tidecv's AP oracle is undefined when a correction removes all GT support.
                delta[error_type.short_name] = None
        counts = Counter(event["type"] for event in events)
        support = Counter(
            exclusive_size_bin(ann["area"])
            for ann in model_coco["annotations"]
            if not ann.get("iscrowd", 0)
        )
        by_size = {}
        for size in SIZE_BINS:
            subset = [event for event in events if event["size_bin"] == size]
            size_counts = Counter(event["type"] for event in subset)
            total = len(subset)
            shifts = sorted(
                event["center_distance_px"] for event in subset if event["type"] == "Loc"
            )
            middle = len(shifts) // 2
            median = (
                None
                if not shifts
                else shifts[middle]
                if len(shifts) % 2
                else (shifts[middle - 1] + shifts[middle]) / 2
            )
            by_size[size] = {
                "gt_support": support[size],
                "total_error_events": total,
                "counts": {kind: size_counts[kind] for kind in ERRORS},
                "percent": {
                    kind: 100 * size_counts[kind] / total if total else 0.0 for kind in ERRORS
                },
                "loc_center_shift_median_px": median,
            }
        report["models"][name] = {
            "predictions": str(path.resolve()),
            "predictions_sha256": sha256(path),
            "verified_manifest": manifest is not None,
            "purpose": None if manifest is None else manifest.get("purpose"),
            "export_settings": None if manifest is None else manifest.get("predict_settings"),
            "prediction_count": len(rows),
            "image_count": len(model_coco["images"]),
            "tide_ap50_percent": float(run.ap),
            "official_ap_percent": {
                size: None
                if official["bins"][size]["AP"] is None
                else 100 * official["bins"][size]["AP"]
                for size in ("all", *SIZE_BINS)
            },
            "delta_ap_pp": {kind: delta[kind] for kind in ERRORS},
            "error_counts": {kind: counts[kind] for kind in ERRORS},
            "size_bins": by_size,
        }
    present_sets = [image_set for image_set in image_sets if image_set is not None]
    if present_sets and any(image_set != present_sets[0] for image_set in present_sets):
        raise ValueError("Compared models need identical verified image sets")
    if len(present_sets) != len(models):
        report["comparability_warnings"].append(
            "One or more exports lack a manifest; complete image coverage cannot be verified."
        )
        if len(models) > 1 and present_sets and len(present_sets[0]) != len(coco["images"]):
            raise ValueError(
                "A partial image-set comparison requires a verified manifest for every model"
            )
    present_settings = [item for item in settings if item is not None]
    for key in ("imgsz", "conf", "max_det", "augment"):
        if len({json.dumps(item.get(key), sort_keys=True) for item in present_settings}) > 1:
            report["comparability_warnings"].append(
                f"Export setting {key} differs between models; interpret comparisons cautiously."
            )
    for name, data in report["models"].items():
        settings_for_model = data["export_settings"]
        if data["purpose"] == "smoke_only":
            report["comparability_warnings"].append(
                f"{name} is a smoke-only export and must not be presented as a full benchmark."
            )
        if settings_for_model and settings_for_model.get("conf", 0) > protocol["export_confidence"]:
            report["comparability_warnings"].append(
                f"{name} uses export confidence {settings_for_model['conf']}; lower-score errors are unavailable."
            )
    _render_charts(report, output_dir)
    if image_root is not None:
        report["gallery_examples"] = _render_gallery(all_events, coco, Path(image_root), output_dir)
        (output_dir / "gallery_examples.json").write_text(
            json.dumps(report["gallery_examples"], indent=2), encoding="utf-8"
        )
    (output_dir / "failure_analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report
