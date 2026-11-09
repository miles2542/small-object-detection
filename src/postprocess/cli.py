"""Typer CLI interface for post-processing, TTA, and multi-model fusion."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from typer.core import TyperCommand

from src.postprocess.ensemble import ensemble_predictions
from src.postprocess.tta import TestTimeAugmentor
from src.utils import ui


class MultiValueOptionCommand(TyperCommand):
    """Command that allows multi-value options such as --weights 1.0 1.0."""

    def parse_args(self, ctx, args):
        new_args = []
        i = 0
        while i < len(args):
            arg = args[i]
            new_args.append(arg)
            i += 1
            if arg in ("--weights", "-w", "--scales"):
                if i < len(args) and not args[i].startswith("-"):
                    new_args.append(args[i])
                    i += 1
                while i < len(args) and not args[i].startswith("-"):
                    try:
                        float(args[i])
                        new_args.extend([arg, args[i]])
                        i += 1
                    except ValueError:
                        break
        return super().parse_args(ctx, new_args)


postprocess_app = typer.Typer(
    help="Post-processing, Test-Time Augmentation (TTA), and Weighted Boxes Fusion.",
    add_completion=False,
)


@postprocess_app.command("fuse", cls=MultiValueOptionCommand)
def postprocess_fuse(
    input_files: list[Path] = typer.Option(
        ...,
        "--input",
        "-i",
        help="Paths to COCO prediction JSON files to ensemble",
    ),
    weights: list[float] | None = typer.Option(
        None,
        "--weights",
        "-w",
        help="Per-model confidence weights",
    ),
    iou_threshold: float = typer.Option(
        0.50,
        "--iou-threshold",
        help="IoU threshold for box clustering",
    ),
    conf_threshold: float = typer.Option(
        0.25,
        "--conf-threshold",
        help="Confidence threshold for fused output detections",
    ),
    conf_type: str = typer.Option(
        "avg",
        "--conf-type",
        help="Consensus mode: 'avg' or 'model_scaled'",
    ),
    output: Path = typer.Option(
        Path("runs/ensemble/fused_predictions.json"),
        "--output",
        "-o",
        help="Destination path for fused COCO prediction JSON",
    ),
) -> None:
    """Fuse multiple COCO prediction JSON files using Weighted Boxes Fusion."""
    try:
        fused = ensemble_predictions(
            prediction_json_paths=input_files,
            weights=weights,
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold,
            conf_type=conf_type,
            output_json=output,
        )
    except (ValueError, TypeError, FileNotFoundError) as error:
        ui.print_error(str(error))
        raise typer.Exit(code=1) from error

    ui.print_header(
        "Weighted Boxes Fusion Ensemble",
        f"Ensembled {len(input_files)} model streams | {len(fused)} fused detections",
    )
    ui.print_success(f"Saved fused predictions to {output}")


@postprocess_app.command("tta")
def postprocess_tta(
    weights: Path = typer.Option(
        ...,
        "--weights",
        "-w",
        exists=True,
        dir_okay=False,
        help="Path to trained model weights checkpoint (.pt)",
    ),
    source: Path = typer.Option(
        ...,
        "--source",
        "-s",
        exists=True,
        help="Path to an image file or directory of images",
    ),
    scales: list[float] | None = typer.Option(
        None,
        "--scales",
        help="Multi-scale factors for TTA (e.g. 0.85 1.0 1.15)",
    ),
    hflip: bool = typer.Option(
        True,
        "--hflip/--no-hflip",
        help="Enable horizontal flip augmentation",
    ),
    vflip: bool = typer.Option(
        False,
        "--vflip/--no-vflip",
        help="Enable vertical flip augmentation",
    ),
    iou_threshold: float = typer.Option(
        0.50,
        "--iou-threshold",
        help="IoU threshold for WBF merging across views",
    ),
    conf_threshold: float = typer.Option(
        0.25,
        "--conf-threshold",
        help="Confidence floor for saving detections",
    ),
    output: Path = typer.Option(
        Path("runs/tta_predictions.json"),
        "--output",
        "-o",
        help="Destination path for TTA predictions JSON",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Inference device ('cpu' or CUDA index)",
    ),
) -> None:
    """Run Test-Time Augmentation inference on images using lazy weight loading."""
    from ultralytics import YOLO

    from src.infer.utils import build_category_mapping

    try:
        model = YOLO(str(weights))
        model.to(device)
    except (OSError, RuntimeError, ValueError) as error:
        ui.print_error(f"Failed to load model weights: {error}")
        raise typer.Exit(code=1) from error

    mapping: dict[int, int] = {}
    if hasattr(model, "names") and isinstance(model.names, (dict, list)):
        try:
            mapping = build_category_mapping(model.names, smoke=True)
        except (ValueError, TypeError, KeyError):
            mapping = {}

    if source.is_file():
        image_paths = [source]
    else:
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        image_paths = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts)

    if not image_paths:
        ui.print_error(f"No valid image files found in {source}")
        raise typer.Exit(code=1)

    augmentor = TestTimeAugmentor(
        scales=scales or [1.0],
        hflip=hflip,
        vflip=vflip,
        iou_threshold=iou_threshold,
        skip_box_threshold=0.001,
    )

    results: list[dict] = []
    progress = ui.create_progress()

    with progress:
        task_id = progress.add_task("Running TTA inference...", total=len(image_paths))
        for idx, img_p in enumerate(image_paths):
            try:
                img_id = int(img_p.stem)
            except ValueError:
                img_id = idx

            f_boxes, f_scores, f_labels = augmentor.forward(model, img_p, device=device)

            for b, s, l in zip(f_boxes, f_scores, f_labels, strict=True):
                if s < conf_threshold:
                    continue
                x1, y1, x2, y2 = b.tolist()
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0.0 or h <= 0.0:
                    continue

                cat_id = mapping.get(int(l), int(l))
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                        "score": round(float(s), 4),
                    }
                )
            progress.update(task_id, advance=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ui.print_header("Test-Time Augmentation", f"{len(image_paths)} images processed | {len(results)} detections")
    ui.print_success(f"Saved TTA predictions to {output}")


@postprocess_app.command("bench")
def postprocess_bench(
    ground_truth: Path = typer.Option(
        ...,
        "--ground-truth",
        "-g",
        exists=True,
        dir_okay=False,
        help="Path to ground truth annotations COCO JSON",
    ),
    predictions: Path = typer.Option(
        ...,
        "--predictions",
        "-p",
        exists=True,
        dir_okay=False,
        help="Path to predictions COCO JSON",
    ),
    output_metrics: Path | None = typer.Option(
        None,
        "--output-metrics",
        "--output-dir",
        "-o",
        help="Optional destination path for metrics JSON file or output directory",
    ),
    confidence: float | None = typer.Option(
        None,
        "--confidence",
        help="Operating-point confidence score",
    ),
) -> None:
    """Evaluate ensembled or TTA predictions against ground truth on AI-TOD micro-scale tiers."""
    from src.eval.coco_eval import evaluate_coco

    try:
        coco = json.loads(ground_truth.read_text(encoding="utf-8"))
        preds = json.loads(predictions.read_text(encoding="utf-8"))
        metrics, _ = evaluate_coco(coco, preds, confidence=confidence)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        ui.print_error(f"Evaluation failed: {error}")
        raise typer.Exit(code=1) from error

    bins = metrics.get("bins", {})
    scale_rows: list[list[str]] = []
    bin_labels = [
        ("all", "All scales"),
        ("very_tiny", "Very Tiny (AP_vt) [0-64 px^2]"),
        ("tiny", "Tiny (AP_t) [64-256 px^2]"),
        ("small", "Small (AP_s) [256-1024 px^2]"),
        ("medium", "Medium (AP_m) [>1024 px^2]"),
    ]

    for key, label in bin_labels:
        b_key = key if key in bins else ("verytiny" if key == "very_tiny" else key)
        if b_key in bins:
            b_data = bins[b_key]
            ap_val = f"{b_data['AP']:.4f}" if b_data.get("AP") is not None else "N/A"
            ap50_val = f"{b_data['AP50']:.4f}" if b_data.get("AP50") is not None else "N/A"
            ar_val = f"{b_data['AR']:.4f}" if b_data.get("AR") is not None else "N/A"
            supp_val = str(b_data.get("support", 0))
            scale_rows.append([label, ap_val, ap50_val, ar_val, supp_val])

    ui.print_header(
        "AI-TOD Micro-Scale Benchmark",
        f"Evaluated {len(metrics.get('image_ids', []))} images against {ground_truth.name}",
    )
    ui.print_table(
        "Scale-Partitioned Performance",
        ["Scale Interval", "AP", "AP50", "AR", "Support"],
        scale_rows,
        column_alignments=["left", "right", "right", "right", "right"],
    )

    if output_metrics is not None:
        if output_metrics.suffix.lower() == ".json":
            save_path = output_metrics
            save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            save_path = output_metrics / "metrics.json"
            output_metrics.mkdir(parents=True, exist_ok=True)

        save_path.write_text(
            json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
        )
        ui.print_success(f"Saved benchmark metrics to {save_path}")
