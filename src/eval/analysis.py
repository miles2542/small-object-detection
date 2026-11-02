"""Compare verified evaluation runs and extract illustrative errors without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.audit import sha256
from src.eval.coco_eval import evaluate_coco, operating_counts

SIZE_BINS = ["verytiny", "tiny", "small", "medium"]


def ap_gap(value: float | None, reference: float | None) -> dict:
    """Signed loss versus a stated reference; preserve improvements and undefined ratios."""
    return {
        "gap_pp": None if value is None or reference is None else 100 * (reference - value),
        "relative_loss_percent": (
            None
            if value is None or reference is None or reference == 0
            else 100 * (1 - value / reference)
        ),
    }


def comparison_markdown(analysis: dict) -> str:
    """Portable numerical handoff; generated from unrounded data, never promotes smoke scores."""

    def number(value, factor=1):
        return "N/A" if value is None else f"{value * factor:.4f}"

    lines = [
        "# Saved-run evaluation comparison",
        "",
        f"**Status:** {analysis['status']}",
        "",
        "AP/AR units: **percent**. These are measured inputs, not proof of training provenance.",
        "",
        "| Model | Images | Area | GT | AP | AP50 | AP75 | AR |",
        "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, data in analysis["models"].items():
        for label, item in data["bins"].items():
            values = " | ".join(number(item[key], 100) for key in ["AP", "AP50", "AP75", "AR"])
            lines.append(
                f"| {name} | {data['image_count']} | {label} | {item['support']} | {values} |"
            )
    lines += [
        "",
        f"## Class-controlled gaps versus {analysis['reference_bin']}",
        "",
        "Signed gap = reference AP minus bin AP; negative is improvement. Relative loss is N/A for zero/missing reference.",
        "",
        "| Model | Class | Area | GT | AP (%) | Gap (pp) | Relative loss (%) |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for name, data in analysis["models"].items():
        for label, gap in data["class_gaps"].items():
            item = data["class_bins"][label]
            lines.append(
                f"| {name} | {data['category_name']} | {label} | {item['support']} | "
                f"{number(item['AP'], 100)} | {number(gap['gap_pp'])} | {number(gap['relative_loss_percent'])} |"
            )
    lines += ["", "## Input identity", ""]
    for name, data in analysis["models"].items():
        provenance = data["provenance"]
        manifest = provenance.get("export_manifest")
        weights = manifest["weights"]["sha256"] if manifest else "unverified external input"
        lines += [
            f"- **{name}:** `{data['run']}`; purpose `{provenance['purpose']}`.",
            f"  Metrics SHA-256: `{data['metrics_sha256']}`.",
            f"  Annotations: `{provenance['annotations_sha256']}`.",
            f"  Predictions: `{provenance['predictions_sha256']}`.",
            f"  Weights: `{weights}`.",
        ]
    lines += [
        "",
        "Inclusive area boundaries overlap; support counts must not be summed as distinct objects.",
        "A small or empty reference can make a ratio unstable or undefined. Inspect support and class composition.",
        "",
    ]
    return "\n".join(lines)


def failure_candidates(evaluator, category_id: int, confidence: float) -> list[dict]:
    """Find unmatched GT at IoU .50 using official matches, then describe nearby boxes.

    A same-class overlap below .50 is a localization candidate, not a causal/TIDE label.
    Crowd and ignored GT are excluded; a prediction matched below the cutoff is absent.
    """
    p = evaluator.params
    k = list(p.catIds).index(category_id)
    t = int(np.flatnonzero(np.isclose(p.iouThrs, 0.5))[0])
    start = (k * len(p.areaRng) + list(p.areaRngLbl).index("all")) * len(p.imgIds)
    records = []
    for entry in evaluator.evalImgs[start : start + len(p.imgIds)]:
        if entry is None:
            continue
        retained = {
            did for did, score in zip(entry["dtIds"], entry["dtScores"]) if score >= confidence
        }
        nearby = sorted(
            (evaluator.cocoDt.anns[did] for did in retained), key=lambda d: (-d["score"], d["id"])
        )
        for j, gid in enumerate(entry["gtIds"]):
            if entry["gtIgnore"][j] or entry["gtMatches"][t, j] in retained:
                continue
            gt = evaluator.cocoGt.anns[gid]
            if not 4 <= gt["area"] <= 256:
                continue
            best, overlap = None, 0.0
            gx, gy, gw, gh = gt["bbox"]
            for detection in nearby:
                x, y, w, h = detection["bbox"]
                intersection = max(0, min(gx + gw, x + w) - max(gx, x)) * max(
                    0, min(gy + gh, y + h) - max(gy, y)
                )
                iou = intersection / (gw * gh + w * h - intersection)
                if iou > overlap:
                    best, overlap = detection, iou
            records.append(
                {
                    "image_id": int(entry["image_id"]),
                    "annotation_id": int(gid),
                    "source_annotation_id": gt.get("source_annotation_id", gid),
                    "bbox": gt["bbox"],
                    "area": gt["area"],
                    "scale_px": gt["area"] ** 0.5,
                    "kind": "localization_candidate" if 0.1 <= overlap < 0.5 else "missed",
                    "best_same_class_iou": overlap,
                    "nearest_prediction": None
                    if best is None
                    else {
                        "bbox": best["bbox"],
                        "score": best["score"],
                        "id": best["id"],
                    },
                }
            )
    return records


def analyze_runs(
    runs: dict[str, Path],
    output_dir: Path,
    *,
    category_id: int | None = None,
    reference_bin: str = "small",
    image_root: Path | None = None,
) -> dict:
    """Validate paired inputs, calculate signed gaps and publish reproducible diagnostics."""
    if len(runs) != 2 or reference_bin not in SIZE_BINS:
        raise ValueError("Provide exactly two labeled runs and a valid reference bin")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use a fresh analysis output directory")
    effective_category_id = category_id
    result = {
        "status": "diagnostic_only",
        "category_id": effective_category_id,
        "reference_bin": reference_bin,
        "models": {},
        "examples": [],
    }
    common, export_signature, all_candidates, coco = None, None, {}, None
    for name, path in runs.items():
        path = Path(path)
        record = json.loads((path / "run.json").read_text(encoding="utf-8"))
        metrics_path = path / "evaluation/metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        provenance = metrics["provenance"]
        annotations, predictions = (
            Path(record["evaluated_annotations"]),
            Path(record["predictions"]),
        )
        if record["status"] != "complete" or any(
            sha256(file) != provenance[key]
            for file, key in [
                (annotations, "annotations_sha256"),
                (predictions, "predictions_sha256"),
            ]
        ):
            raise ValueError("Incomplete run or changed annotation/prediction input")
        signature = (
            provenance["annotations_sha256"],
            metrics["protocol"],
            metrics["image_ids"],
            metrics["category_ids"],
        )
        if common is not None and signature != common:
            raise ValueError(
                "Comparison needs identical annotations, image/category IDs and protocol"
            )
        common = signature
        coco = json.loads(annotations.read_text(encoding="utf-8"))
        rows = json.loads(predictions.read_text(encoding="utf-8"))
        check, evaluator = evaluate_coco(
            coco,
            rows,
            image_ids=metrics["image_ids"],
            confidence=metrics["protocol"]["operating_confidence"],
        )
        if any(check[key] != metrics[key] for key in ["bins", "per_class"]):
            raise ValueError("Saved metrics do not reproduce from their recorded inputs")
        if effective_category_id is None:
            if "1" in metrics["per_class"]:
                effective_category_id = 1
            elif "6" in metrics["per_class"]:
                effective_category_id = 6
            else:
                effective_category_id = int(next(iter(metrics["per_class"])))
            result["category_id"] = effective_category_id
        category = metrics["per_class"][str(effective_category_id)]
        reference = category["bins"][reference_bin]["AP"]
        manifest = provenance.get("export_manifest")
        if manifest:
            if (
                manifest["status"] != "complete"
                or manifest["predictions_sha256"] != provenance["predictions_sha256"]
                or manifest["annotations"]["sha256"] != provenance["annotations_sha256"]
            ):
                raise ValueError("Export manifest disagrees with evaluated input fingerprints")
            settings = manifest["predict_settings"]
            signature = {
                key: settings[key]
                for key in ["imgsz", "conf", "max_det", "augment", "rect", "agnostic_nms", "iou"]
            }
            signature["target_categories"] = sorted(
                entry["category_id"] for entry in manifest["category_mapping"].values()
            )
            if export_signature is not None and signature != export_signature:
                raise ValueError(
                    "Export resolution, floor, cap, augmentation or target vocabulary differs"
                )
            export_signature = signature
        floor = manifest["predict_settings"]["conf"] if manifest else None
        thresholds = [
            score for score in [0.001, 0.01, 0.1, 0.25, 0.5] if floor is not None and score >= floor
        ]
        sweep = []
        for score in thresholds:
            counts = operating_counts(evaluator, 0, score)["per_class"][str(effective_category_id)]
            sweep.append({"score": score, **{key: value[0] for key, value in counts.items()}})
        confidence = metrics["protocol"]["operating_confidence"]
        all_candidates[name] = failure_candidates(evaluator, effective_category_id, confidence)
        result["models"][name] = {
            "run": path.as_posix(),
            "metrics_sha256": sha256(metrics_path),
            "provenance": provenance,
            "protocol": metrics["protocol"],
            "image_count": len(metrics["image_ids"]),
            "category_name": category["name"],
            "bins": metrics["bins"],
            "class_bins": category["bins"],
            "class_gaps": {
                label: ap_gap(category["bins"][label]["AP"], reference) for label in SIZE_BINS
            },
            "confidence_sweep": sweep,
            "candidate_counts": {
                kind: sum(r["kind"] == kind for r in all_candidates[name])
                for kind in ["missed", "localization_candidate"]
            },
        }
    # Select one example of each kind, anchored to the first run; deterministic, no manual cherry-picking.
    first_name = next(iter(runs))
    for kind in ["missed", "localization_candidate"]:
        candidates = [r for r in all_candidates[first_name] if r["kind"] == kind]
        if candidates:
            chosen = min(candidates, key=lambda r: (r["image_id"], r["annotation_id"]))
            chosen = {**chosen, "anchor_model": first_name, "by_model": {}}
            for name, records in all_candidates.items():
                chosen["by_model"][name] = next(
                    (r for r in records if r["annotation_id"] == chosen["annotation_id"]), None
                )
            result["examples"].append(chosen)
    result["all_runs_verified_exports"] = all(
        item["provenance"]["export_manifest"] is not None for item in result["models"].values()
    )
    result["contains_smoke"] = any(
        item["provenance"]["purpose"] == "smoke_only" for item in result["models"].values()
    )
    result["status"] = (
        "smoke_diagnostic_only"
        if result["contains_smoke"]
        else "training_provenance_review_required"
    )
    result["analysis_code_sha256"] = sha256(Path(__file__))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "comparison.md").write_text(comparison_markdown(result), encoding="utf-8")
    from src.eval.plots import plot_analysis_comparison, plot_failure_examples

    plot_analysis_comparison(result, output_dir / "size_comparison.png", gaps=False)
    plot_analysis_comparison(result, output_dir / "size_gaps.png", gaps=True)
    if image_root is not None and result["examples"]:
        plot_failure_examples(result, coco, Path(image_root), output_dir / "failure_examples.png")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, help="LABEL=CLI_RUN_DIRECTORY (twice)"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--reference-bin", choices=SIZE_BINS, default="small")
    args = parser.parse_args()
    try:
        pairs = [item.split("=", 1) for item in args.run]
        if any(len(pair) != 2 for pair in pairs) or len({p[0] for p in pairs}) != len(pairs):
            raise ValueError("Use distinct LABEL=CLI_RUN_DIRECTORY entries")
        result = analyze_runs(
            {name: Path(path) for name, path in pairs},
            args.output_dir,
            category_id=args.category_id,
            reference_bin=args.reference_bin,
            image_root=args.image_root,
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(f"Analysis saved: {args.output_dir} | {result['status']}")


if __name__ == "__main__":
    main()
