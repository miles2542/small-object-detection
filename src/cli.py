"""Modular CLI interface for the Small Object Detection benchmarking suite."""

from __future__ import annotations

import json
import math
from pathlib import Path

import typer

from src.utils import ui

app = typer.Typer(
    name="sod",
    help="Small Object Detection (AI-TOD-v2 / YOLOv8 / YOLO26) Benchmarking Suite.",
    add_completion=False,
)

data_app = typer.Typer(help="Dataset management, conversion, EDA, and validation.")
app.add_typer(data_app, name="data")

train_app = typer.Typer(help="Training execution across baselines and ablations.")
app.add_typer(train_app, name="train")

bench_app = typer.Typer(help="Runtime latency, throughput, and VRAM benchmarking.")
app.add_typer(bench_app, name="bench")

kaggle_app = typer.Typer(help="Headless Kaggle GPU training orchestration.")
bench_app.add_typer(kaggle_app, name="kaggle")

eval_app = typer.Typer(help="Dual-surface evaluation on AI-TOD micro-scale tiers.")
app.add_typer(eval_app, name="eval")

infer_app = typer.Typer(help="Inference strategies: High-resolution and SAHI.")
app.add_typer(infer_app, name="infer")

model_app = typer.Typer(help="Model architecture inspection and verification.")
app.add_typer(model_app, name="model")


@app.callback()
def main_callback() -> None:
    """Entry point for Small Object Detection CLI."""


@eval_app.command("run")
def eval_run(
    data: Path = typer.Option(
        ...,
        "--data",
        exists=True,
        dir_okay=False,
        help="Dataset YAML for weights, or exact COCO JSON for saved predictions",
    ),
    weights: Path | None = typer.Option(None, "--weights", exists=True, dir_okay=False),
    predictions: Path | None = typer.Option(None, "--predictions", exists=True, dir_okay=False),
    split: str = typer.Option(
        "val", "--split", help="train, val or test; JSON defines its own image set"
    ),
    output_dir: Path = typer.Option(Path("runs/eval/result"), "--output-dir", "-o"),
    annotations: Path | None = typer.Option(None, "--annotations", exists=True, dir_okay=False),
    image_root: Path | None = typer.Option(None, "--image-root", exists=True, file_okay=False),
    smoke: bool = typer.Option(
        False, "--smoke", help="Explicitly allow pretrained/partial-vocabulary diagnostics"
    ),
    device: str = typer.Option("cpu", "--device"),
    batch_size: int = typer.Option(4, "--batch-size", min=1),
    confidence: float | None = typer.Option(
        None,
        "--confidence",
        min=0,
        max=1,
        help="Operating-point score only; AP ranking stays unchanged",
    ),
) -> None:
    """Evaluate COCO predictions or export and evaluate a native .pt checkpoint."""
    from src.eval.runner import run_evaluation

    try:
        metrics = run_evaluation(
            data,
            output_dir,
            weights=weights,
            predictions=predictions,
            split=split,
            annotations=annotations,
            image_root=image_root,
            smoke=smoke,
            device=device,
            batch_size=batch_size,
            confidence=confidence,
        )
    except (ValueError, TypeError, KeyError, OSError) as error:
        ui.print_error(str(error))
        raise typer.Exit(code=1) from error
    purpose = metrics["provenance"]["purpose"]
    ui.print_header(
        "AI-TOD size-based evaluation",
        f"{purpose} | {len(metrics['image_ids'])} images | {metrics['protocol']['name']}",
    )
    if purpose != "evaluation":
        ui.print_warning(
            "SMOKE ONLY - not trained AI-TOD baseline metrics"
            if purpose == "smoke_only"
            else "External predictions: training/export provenance is unverified"
        )

    def percent(value):
        return "N/A" if value is None else f"{100 * value:.4f}"

    ui.print_table(
        "AP / AR (%)",
        ["Area", "GT", "AP", "AP50", "AP75", "AR"],
        [
            [
                label,
                str(values["support"]),
                *[percent(values[k]) for k in ["AP", "AP50", "AP75", "AR"]],
            ]
            for label, values in metrics["bins"].items()
        ],
    )
    ui.print_table(
        "Fixed-confidence counts at IoU 0.50",
        ["Area", "TP", "FP", "FN", "P (%)", "R (%)"],
        [
            [
                label,
                *[str(values["operating_point"]["micro"][k][0]) for k in ["tp", "fp", "fn"]],
                *[
                    percent(values["operating_point"]["micro"][k][0])
                    for k in ["precision", "recall"]
                ],
            ]
            for label, values in metrics["bins"].items()
        ],
    )
    ui.print_success(
        f"Results: {output_dir / 'evaluation'} | operating score: {metrics['protocol']['operating_confidence']}"
    )


@eval_app.command("compare")
def eval_compare(
    runs: list[str] = typer.Option(
        ...,
        "--run",
        "-r",
        help="Run specifiers in LABEL=RUN_DIRECTORY format (e.g. --run baseline=runs/eval/yolov8s --run exp=runs/eval/sahi)",
    ),
    output_dir: Path = typer.Option(
        Path("runs/eval/comparison"),
        "--output-dir",
        "-o",
        help="Destination directory for comparison report, figures, and analysis",
    ),
    category_id: int | None = typer.Option(
        None,
        "--category-id",
        "-c",
        help="Target category ID for class-controlled scale gap analysis (defaults to 1 if present, else first available)",
    ),
    reference_bin: str = typer.Option(
        "small",
        "--reference-bin",
        help="Reference size bin for AP gap calculation (verytiny, tiny, small, medium)",
    ),
    image_root: Path | None = typer.Option(
        None,
        "--image-root",
        exists=True,
        file_okay=False,
        help="Optional image directory to render qualitative failure visualizations",
    ),
) -> None:
    """Compare evaluation runs, compute scale-controlled gaps, and output diagnostic reports."""
    from src.eval.analysis import analyze_runs

    pairs = [item.split("=", 1) for item in runs]
    if any(len(pair) != 2 or not pair[0] or not pair[1] for pair in pairs):
        ui.print_error("Each --run must be in LABEL=RUN_DIRECTORY format (e.g. --run baseline=runs/eval/base)")
        raise typer.Exit(code=1)
    if len({p[0] for p in pairs}) != len(pairs):
        ui.print_error("Run labels must be distinct")
        raise typer.Exit(code=1)
    run_dict = {name: Path(path) for name, path in pairs}

    try:
        analysis = analyze_runs(
            run_dict,
            output_dir,
            category_id=category_id,
            reference_bin=reference_bin,
            image_root=image_root,
        )
    except (ValueError, TypeError, KeyError, OSError) as error:
        ui.print_error(str(error))
        raise typer.Exit(code=1) from error

    cat_id = analysis["category_id"]
    ui.print_header(
        "AI-TOD Evaluation Comparison",
        f"Status: {analysis['status']} | Ref bin: {analysis['reference_bin']} | Target class: ID {cat_id}",
    )

    def percent(value):
        return "N/A" if value is None else f"{100 * value:.4f}"

    def gap_fmt(value):
        return "N/A" if value is None else f"{value:+.4f}"

    headers = ["Model", "Area", "GT", "AP (%)", "AP50 (%)", "AP75 (%)", "AR (%)"]
    rows = []
    for model_name, data in analysis["models"].items():
        for label, item in data["bins"].items():
            rows.append([
                model_name,
                label,
                str(item["support"]),
                percent(item["AP"]),
                percent(item["AP50"]),
                percent(item["AP75"]),
                percent(item["AR"]),
            ])
    ui.print_table("Model Comparison Across Scale Bins", headers, rows)

    first_model = next(iter(analysis["models"].values()))
    cat_name = first_model.get("category_name", f"Class {cat_id}")
    gap_headers = ["Model", "Class", "Area", "GT", "AP (%)", f"Gap vs {reference_bin} (pp)", "Rel Loss (%)"]
    gap_rows = []
    for model_name, data in analysis["models"].items():
        c_name = data.get("category_name", str(cat_id))
        for label, gap in data["class_gaps"].items():
            item = data["class_bins"][label]
            gap_rows.append([
                model_name,
                c_name,
                label,
                str(item["support"]),
                percent(item["AP"]),
                gap_fmt(gap["gap_pp"]),
                gap_fmt(gap["relative_loss_percent"]),
            ])
    ui.print_table(f"Class-Controlled Scale Gaps ({cat_name} vs {reference_bin})", gap_headers, gap_rows)

    ui.print_success(
        f"Comparison saved to {output_dir / 'analysis.json'} | report: {output_dir / 'comparison.md'}"
    )


@infer_app.command("sahi")
def infer_sahi(
    weights: Path = typer.Option(
        ..., "--weights", exists=True, dir_okay=False, help="Path to model weights (.pt)"
    ),
    data: Path = typer.Option(
        ..., "--data", exists=True, dir_okay=False, help="Dataset YAML or COCO ground-truth JSON"
    ),
    output_dir: Path = typer.Option(
        Path("runs/infer/sahi"),
        "--output-dir",
        "-o",
        help="Destination directory for predictions and manifest",
    ),
    split: str = typer.Option("val", "--split", help="Dataset split (train, val, test)"),
    slice_size: int = typer.Option(
        512, "--slice-size", min=64, help="Patch slice height and width in pixels"
    ),
    overlap: float = typer.Option(
        0.20, "--overlap", min=0.0, max=0.9, help="Overlap ratio between adjacent slices"
    ),
    iou: float = typer.Option(
        0.60, "--iou", min=0.0, max=1.0, help="IoU merging threshold for post-slice NMS/NMM"
    ),
    postprocess: str = typer.Option(
        "GREEDYNMM",
        "--postprocess",
        help="Postprocessing merge strategy (GREEDYNMM, NMS, NMM)",
    ),
    full_frame: bool = typer.Option(
        True,
        "--full-frame/--no-full-frame",
        help="Include full-frame inference alongside patches",
    ),
    confidence: float = typer.Option(
        0.001, "--confidence", min=0.0, max=1.0, help="Confidence threshold"
    ),
    device: str = typer.Option("cpu", "--device", help="Computation device (cpu, 0, cuda:0)"),
    smoke: bool = typer.Option(
        False, "--smoke", help="Permit partial-vocabulary diagnostics"
    ),
    sample_limit: int | None = typer.Option(
        None, "--sample-limit", min=1, help="Limit sample count for quick runs"
    ),
    annotations: Path | None = typer.Option(
        None, "--annotations", exists=True, dir_okay=False, help="Explicit COCO annotations JSON"
    ),
    image_root: Path | None = typer.Option(
        None, "--image-root", exists=True, file_okay=False, help="Explicit image root folder"
    ),
) -> None:
    """Run Slicing-Aided Hyper Inference (SAHI) on a dataset split and export standard predictions."""
    from src.infer.sahi import SahiConfig, export_sahi_predictions

    config = SahiConfig(
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        perform_standard_pred=full_frame,
        postprocess_type=postprocess,
        postprocess_match_threshold=iou,
        confidence_threshold=confidence,
    )

    try:
        manifest = export_sahi_predictions(
            weights=weights,
            data=data,
            output_dir=output_dir,
            split=split,
            config=config,
            annotations=annotations,
            image_root=image_root,
            device=device,
            smoke=smoke,
            sample_limit=sample_limit,
        )
    except (ValueError, TypeError, KeyError, OSError, FileNotFoundError) as error:
        ui.print_error(str(error))
        raise typer.Exit(code=1) from error

    ui.print_header(
        "SAHI Sliced Inference",
        f"{manifest['total_images']} images | slice {slice_size}x{slice_size} | overlap {overlap:.2f}",
    )
    lat = manifest["latency"]
    ui.print_table(
        "SAHI Execution Profile",
        ["Metric", "Value"],
        [
            ["Images Processed", str(manifest["total_images"])],
            ["Total Detections", str(manifest["total_detections"])],
            ["Zero-Detection Images", str(len(manifest["zero_detection_image_ids"]))],
            ["Mean Slicing Latency", f"{lat['mean_slice_ms']:.2f} ms"],
            ["Mean Model Inference", f"{lat['mean_prediction_ms']:.2f} ms"],
            ["Mean Postprocess / NMS", f"{lat['mean_postprocess_ms']:.2f} ms"],
            ["Mean Total Latency", f"{lat['mean_total_ms']:.2f} ms"],
            ["Throughput", f"{lat['throughput_fps']:.2f} FPS"],
        ],
    )
    ui.print_success(f"Predictions saved to {output_dir / 'predictions.json'}")
    ui.print_success(f"Run manifest saved to {output_dir / 'manifest.json'}")


@infer_app.command("run")
def infer_run(
    weights: Path = typer.Option(
        ..., "--weights", exists=True, dir_okay=False, help="Path to model weights (.pt)"
    ),
    data: Path = typer.Option(
        ..., "--data", exists=True, dir_okay=False, help="Dataset YAML or COCO ground-truth JSON"
    ),
    output_dir: Path = typer.Option(
        Path("runs/infer/high_res"), "--output-dir", "-o", help="Destination directory"
    ),
    split: str = typer.Option("val", "--split", help="Dataset split (train, val, test)"),
    img_size: int = typer.Option(
        800, "--img-size", min=32, help="Input resolution (e.g. 800, 1280, 1600)"
    ),
    confidence: float = typer.Option(
        0.001, "--confidence", min=0.0, max=1.0, help="Confidence threshold"
    ),
    iou: float = typer.Option(0.70, "--iou", min=0.0, max=1.0, help="NMS IoU threshold"),
    device: str = typer.Option("cpu", "--device", help="Computation device (cpu, 0, cuda:0)"),
    smoke: bool = typer.Option(
        False, "--smoke", help="Permit partial-vocabulary diagnostics"
    ),
    sample_limit: int | None = typer.Option(
        None, "--sample-limit", min=1, help="Limit sample count for quick runs"
    ),
    annotations: Path | None = typer.Option(
        None, "--annotations", exists=True, dir_okay=False, help="Explicit COCO annotations JSON"
    ),
    image_root: Path | None = typer.Option(
        None, "--image-root", exists=True, file_okay=False, help="Explicit image root folder"
    ),
) -> None:
    """Run standard full-image high-resolution inference on a dataset split."""
    from src.infer.high_res import HighResConfig, export_high_res_predictions

    config = HighResConfig(
        image_size=img_size,
        confidence_threshold=confidence,
        iou_threshold=iou,
    )

    try:
        manifest = export_high_res_predictions(
            weights=weights,
            data=data,
            output_dir=output_dir,
            split=split,
            image_size=img_size,
            config=config,
            annotations=annotations,
            image_root=image_root,
            device=device,
            smoke=smoke,
            sample_limit=sample_limit,
        )
    except (ValueError, TypeError, KeyError, OSError, FileNotFoundError) as error:
        ui.print_error(str(error))
        raise typer.Exit(code=1) from error

    ui.print_header(
        "Standard High-Resolution Inference",
        f"{manifest['total_images']} images | resolution {img_size}x{img_size}",
    )
    lat = manifest["latency"]
    mem = manifest["memory"]
    ui.print_table(
        "High-Resolution Execution Profile",
        ["Metric", "Value"],
        [
            ["Images Processed", str(manifest["total_images"])],
            ["Total Detections", str(manifest["total_detections"])],
            ["Zero-Detection Images", str(len(manifest["zero_detection_image_ids"]))],
            ["Mean Preprocessing Latency", f"{lat['mean_preprocess_ms']:.2f} ms"],
            ["Mean Forward Inference", f"{lat['mean_inference_ms']:.2f} ms"],
            ["Mean Postprocess / NMS", f"{lat['mean_postprocess_ms']:.2f} ms"],
            ["Mean Total Latency", f"{lat['mean_total_ms']:.2f} ms"],
            ["Throughput", f"{lat['throughput_fps']:.2f} FPS"],
            ["Peak Memory", f"{mem['peak_mb']:.2f} MB"],
        ],
    )
    ui.print_success(f"Predictions saved to {output_dir / 'predictions.json'}")
    ui.print_success(f"Run manifest saved to {output_dir / 'manifest.json'}")


@infer_app.command("bench")
def infer_bench(
    weights: Path = typer.Option(
        ..., "--weights", exists=True, dir_okay=False, help="Path to model weights (.pt)"
    ),
    data: Path = typer.Option(
        Path("configs/aitod_mini.yaml"),
        "--data",
        exists=True,
        dir_okay=False,
        help="Dataset YAML",
    ),
    output_dir: Path = typer.Option(
        Path("runs/infer/latency_bench"),
        "--output-dir",
        "-o",
        help="Destination directory",
    ),
    resolutions: str = typer.Option(
        "800,1280,1600", "--resolutions", help="Comma-separated image resolutions"
    ),
    with_sahi: bool = typer.Option(
        False,
        "--with-sahi/--no-sahi",
        help="Include SAHI sliced inference in benchmark comparison",
    ),
    slice_size: int = typer.Option(512, "--slice-size", min=64, help="SAHI patch size"),
    overlap: float = typer.Option(
        0.20, "--overlap", min=0.0, max=0.9, help="SAHI patch overlap ratio"
    ),
    split: str = typer.Option("val", "--split", help="Dataset split"),
    sample_limit: int | None = typer.Option(
        None, "--sample-limit", min=1, help="Limit sample count for fast benchmarking"
    ),
    device: str = typer.Option("cpu", "--device", help="Computation device"),
    smoke: bool = typer.Option(
        False, "--smoke", help="Permit partial-vocabulary diagnostics"
    ),
    annotations: Path | None = typer.Option(
        None, "--annotations", exists=True, dir_okay=False, help="Explicit COCO annotations JSON"
    ),
    image_root: Path | None = typer.Option(
        None, "--image-root", exists=True, file_okay=False, help="Explicit image root folder"
    ),
) -> None:
    """Benchmark latency, memory, and throughput trade-offs across resolutions and SAHI."""
    from src.infer.high_res import benchmark_resolutions
    from src.infer.sahi import SahiConfig, export_sahi_predictions

    try:
        res_list = [int(r.strip()) for r in resolutions.split(",") if r.strip()]
    except ValueError as err:
        ui.print_error(
            f"Invalid resolutions argument '{resolutions}': expected comma-separated integers"
        )
        raise typer.Exit(code=1) from err

    try:
        summary = benchmark_resolutions(
            weights=weights,
            data=data,
            output_dir=output_dir,
            resolutions=res_list,
            split=split,
            annotations=annotations,
            image_root=image_root,
            device=device,
            smoke=smoke,
            sample_limit=sample_limit,
        )

        if with_sahi:
            sahi_dir = output_dir / f"sahi_{slice_size}_ov{int(overlap * 100)}"
            sahi_cfg = SahiConfig(
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
            )
            sahi_manifest = export_sahi_predictions(
                weights=weights,
                data=data,
                output_dir=sahi_dir,
                split=split,
                config=sahi_cfg,
                annotations=annotations,
                image_root=image_root,
                device=device,
                smoke=smoke,
                sample_limit=sample_limit,
            )
            summary["sahi"] = {
                "strategy": "sahi",
                "slice_size": slice_size,
                "overlap": overlap,
                "latency": sahi_manifest["latency"],
                "total_detections": sahi_manifest["total_detections"],
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "resolution_benchmark.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
    except (ValueError, TypeError, KeyError, OSError, FileNotFoundError) as error:
        ui.print_error(str(error))
        raise typer.Exit(code=1) from error

    ui.print_header(
        "Inference Strategy & Latency Benchmark",
        f"Device: {device} | Evaluated samples: {summary['sample_count']}",
    )

    headers = [
        "Strategy / Resolution",
        "Latency (ms)",
        "Throughput (FPS)",
        "Peak Mem (MB)",
        "Detections",
    ]
    rows = []
    for res_str, res_data in summary["resolutions"].items():
        lat = res_data["latency"]
        mem = res_data["memory"]
        rows.append([
            f"Full Frame ({res_str}x{res_str})",
            f"{lat['mean_total_ms']:.2f}",
            f"{lat['throughput_fps']:.2f}",
            f"{mem['peak_mb']:.2f}",
            str(res_data["total_detections"]),
        ])

    if "sahi" in summary:
        s_data = summary["sahi"]
        s_lat = s_data["latency"]
        rows.append([
            f"SAHI Sliced ({s_data['slice_size']}x{s_data['slice_size']}, ov {s_data['overlap']:.2f})",
            f"{s_lat['mean_total_ms']:.2f}",
            f"{s_lat['throughput_fps']:.2f}",
            "N/A",
            str(s_data["total_detections"]),
        ])

    ui.print_table(
        "Latency & Throughput Trade-off Summary",
        headers,
        rows,
        column_alignments=["left", "right", "right", "right", "right"],
    )
    ui.print_success(f"Benchmark summary saved to {output_dir / 'resolution_benchmark.json'}")


@bench_app.command("runtime")
def bench_runtime(
    config: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
    models: list[str] | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model architectures to benchmark (e.g. yolov8s, yolo26s)",
    ),
    batch_sizes: list[int] | None = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Batch sizes to benchmark (e.g. 1 4 8 16)",
    ),
    output_dir: Path = typer.Option(
        Path(".scratch/benchmarks"),
        "--output-dir",
        "-o",
        help="Directory to save raw benchmark JSON",
    ),
    img_size: int = typer.Option(
        800,
        "--img-size",
        "-s",
        help="Input image resolution (AI-TOD standard: 800)",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Execution device (e.g. cpu, directml, cuda:0)",
    ),
) -> None:
    """Benchmark inference latency, FPS throughput, VRAM, and gradient backward pass."""
    from src.bench.runtime import run_benchmarks

    models_list = models or ["yolov8s", "yolo26s"]
    batches_list = batch_sizes or [1, 4, 8, 16]

    run_benchmarks(
        models=models_list,
        batch_sizes=batches_list,
        img_size=img_size,
        output_dir=output_dir,
        device=device,
    )


@kaggle_app.command("scaffold")
def kaggle_scaffold(
    kernel_dir: Path = typer.Option(
        Path("kaggle/kernels/aitod_train"),
        "--kernel-dir",
        "-k",
        help="Directory to store Kaggle kernel configuration and runner script",
    ),
    kernel_id: str = typer.Option(
        "miles2542/aitod-baseline-training",
        "--kernel-id",
        help="Kaggle kernel identifier (username/slug)",
    ),
    enable_gpu: bool = typer.Option(
        True,
        "--gpu/--no-gpu",
        help="Target Dual NVIDIA T4 GPU accelerator",
    ),
    epochs: int = typer.Option(
        10,
        "--epochs",
        help="Training epochs for remote execution",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        help="Batch size for GPU dataloader",
    ),
) -> None:
    """Generate self-contained headless Kaggle runner scripts and metadata."""
    from src.bench.kaggle_runner import generate_kernel_scaffold

    generate_kernel_scaffold(
        kernel_dir=kernel_dir,
        kernel_id=kernel_id,
        enable_gpu=enable_gpu,
        epochs=epochs,
        batch_size=batch_size,
    )


@kaggle_app.command("push")
def kaggle_push(
    kernel_dir: Path = typer.Option(
        Path("kaggle/kernels/aitod_train"),
        "--kernel-dir",
        "-k",
        help="Directory containing kernel-metadata.json and runner.py",
    ),
) -> None:
    """Push kernel job to Kaggle to trigger remote Dual T4 training."""
    from src.bench.kaggle_runner import push_kernel

    push_kernel(kernel_dir=kernel_dir)


@kaggle_app.command("status")
def kaggle_status(
    kernel_id: str = typer.Option(
        "miles2542/aitod-baseline-training",
        "--kernel-id",
        help="Kaggle kernel identifier",
    ),
) -> None:
    """Check status of a running or queued Kaggle kernel."""
    from src.bench.kaggle_runner import get_kernel_status

    get_kernel_status(kernel_id=kernel_id)


@kaggle_app.command("output")
def kaggle_output(
    kernel_id: str = typer.Option(
        "miles2542/aitod-baseline-training",
        "--kernel-id",
        help="Kaggle kernel identifier",
    ),
    output_dir: Path = typer.Option(
        Path("runs/kaggle"),
        "--output-dir",
        "-o",
        help="Local directory to store downloaded weights and logs",
    ),
) -> None:
    """Download weights and output logs from completed Kaggle kernel."""
    from src.bench.kaggle_runner import download_kernel_output

    download_kernel_output(kernel_id=kernel_id, output_dir=output_dir)


@train_app.command("run")
def train_run(
    model: str = typer.Option(
        "yolov8s",
        "--model",
        "-m",
        help="Model architecture (yolov8s, yolo26s) or weights path",
    ),
    config: Path = typer.Option(
        Path("configs/aitod_full.yaml") if Path("configs/aitod_full.yaml").exists() else Path("configs/aitod_mini.yaml"),
        "--config",
        "-c",
        help="Path to dataset YAML configuration (defaults to full if present)",
    ),
    epochs: int = typer.Option(
        10,
        "--epochs",
        "-e",
        help="Number of training epochs",
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Batch size for dataloader (auto-defaults to 16 on GPU, 4 on CPU)",
    ),
    img_size: int = typer.Option(
        800,
        "--img-size",
        "-s",
        help="Input image resolution in pixels (AI-TOD standard: 800)",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        "-d",
        help="Execution device (auto-detects CUDA GPU if available, else CPU)",
    ),
    save_dir: Path | None = typer.Option(
        None,
        "--save-dir",
        help="Destination directory for checkpoints and artifacts",
    ),
    fraction: float = typer.Option(
        1.0,
        "--fraction",
        help="Fraction of dataset to train on (e.g. 0.01 for fast sanity check)",
    ),
    workers: int = typer.Option(
        2,
        "--workers",
        help="Number of dataloader workers (default: 2)",
    ),
    pretrained: str | None = typer.Option(
        None,
        "--pretrained",
        help="Checkpoint to transfer matching weights from when --model is an architecture YAML",
    ),
    nwd: bool = typer.Option(
        False,
        "--nwd/--no-nwd",
        help="Train with the hybrid CIoU/NWD box loss instead of CIoU",
    ),
    nwd_ratio: float = typer.Option(
        0.5,
        "--nwd-ratio",
        min=0.0,
        max=1.0,
        help="CIoU weight alpha of the hybrid box loss (0 = pure NWD, 1 = pure CIoU)",
    ),
    nwd_constant: float = typer.Option(
        12.8,
        "--nwd-constant",
        min=1e-6,
        help="NWD normalization constant C in pixels",
    ),
) -> None:
    """Train a detection model (YOLOv8s, YOLO26s or a P2 architecture YAML) on a dataset config."""
    from src.models.train import train_baseline

    train_baseline(
        model=model,
        config=config,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        device=device,
        save_dir=save_dir,
        fraction=fraction,
        workers=workers,
        pretrained=pretrained,
        nwd=nwd,
        nwd_ratio=nwd_ratio,
        nwd_constant=nwd_constant,
    )


@model_app.command("info")
def model_info(
    config: Path = typer.Option(..., "--config", "-c", help="Model architecture YAML"),
    imgsz: int = typer.Option(800, "--imgsz", help="Input resolution used for the measurements"),
    num_classes: int = typer.Option(8, "--num-classes", help="Number of detection classes"),
) -> None:
    """Print architecture summary: head count, strides, parameters, FLOPs and receptive fields."""
    from src.models.builder import build_model, measure_receptive_fields, verify_model

    model = build_model(config, num_classes=num_classes)
    report = verify_model(model, imgsz=imgsz)
    fields = measure_receptive_fields(model, imgsz=imgsz)

    ui.print_header("Model Info", str(config))
    ui.print_table(
        title="Architecture",
        columns=["Property", "Value"],
        rows=[
            ["Detection heads", str(report.head_count)],
            ["Strides", ", ".join(str(s) for s in report.strides)],
            ["End-to-end head", "yes" if report.end2end else "no"],
            ["Classes", str(report.num_classes)],
            ["Parameters", f"{report.parameters:,}"],
            [f"GFLOPs @ {imgsz}", f"{report.gflops:.1f}"],
        ],
    )
    ui.print_table(
        title="Per-scale head inputs",
        columns=["Stride", "Feature map", "Receptive field (px)"],
        rows=[
            [str(s), "x".join(str(d) for d in shape[1:]), str(rf)]
            for s, shape, rf in zip(report.strides, report.feature_shapes, fields, strict=True)
        ],
    )


@model_app.command("verify")
def model_verify(
    config: Path = typer.Option(..., "--config", "-c", help="Model architecture YAML"),
    imgsz: int = typer.Option(800, "--imgsz", help="Dummy input resolution"),
    num_classes: int = typer.Option(8, "--num-classes", help="Number of detection classes"),
) -> None:
    """Run a dummy forward pass and check feature-map shapes against the head strides."""
    from src.models.builder import build_model, verify_model

    model = build_model(config, num_classes=num_classes)
    try:
        report = verify_model(model, imgsz=imgsz)
    except ValueError as exc:
        ui.print_error(str(exc))
        raise typer.Exit(code=1) from exc

    names = [f"P{round(math.log2(s))}" for s in report.strides]
    ui.print_table(
        title=f"Forward pass ({imgsz}x{imgsz})",
        columns=["Scale", "Stride", "Feature map (N,C,H,W)"],
        rows=[
            [name, str(s), str(shape)]
            for name, s, shape in zip(names, report.strides, report.feature_shapes, strict=True)
        ],
    )
    ui.print_success(f"{report.head_count}-scale head verified at {imgsz}x{imgsz}")


@data_app.command("summary")
def data_summary(
    config: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
) -> None:
    """Display configured dataset paths, classes, and micro-scale bin intervals."""
    import yaml

    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg.get("dataset", {})
    bins = ds_cfg.get("micro_bins", {})
    rows = [[k, f"{v[0]} - {v[1]} px"] for k, v in bins.items()]
    ui.print_table("Micro-scale Bins (AI-TOD)", ["Bin", "Pixel Area Interval"], rows)


@data_app.command("download")
def data_download(
    dataset: str = typer.Option(
        "visdrone",
        "--dataset",
        help="Dataset to download (visdrone, aitod-v2, all)",
    ),
    extract_dir: Path = typer.Option(
        Path("data"),
        "--extract-dir",
        help="Root directory for dataset storage",
    ),
    splits: list[str] | None = typer.Option(
        None,
        "--splits",
        help="Splits to download (e.g. val train test-dev)",
    ),
    sample_only: bool = typer.Option(
        False,
        "--sample-only",
        help="Extract a small sample set of images for immediate benchmarking",
    ),
    sample_count: int = typer.Option(
        500,
        "--sample-count",
        help="Number of sample images to collect when --sample-only is active",
    ),
) -> None:
    """Download and extract benchmark datasets (VisDrone, AI-TOD-v2)."""
    from src.data.download import download_datasets

    download_datasets(
        dataset=dataset,
        extract_dir=extract_dir,
        splits=splits or ["val"],
        sample_only=sample_only,
        sample_count=sample_count,
    )


@data_app.command("eda")
def data_eda(
    data_dir: Path = typer.Option(
        Path("data/ai-tod-v2/annotations"),
        "--data-dir",
        help="Path to annotation files directory",
    ),
    figures_dir: Path = typer.Option(
        Path("assets/eda"),
        "--figures-dir",
        help="Output directory for diagnostic plots",
    ),
    report_path: Path = typer.Option(
        Path("docs/dataset_analysis.md"),
        "--report-path",
        help="Output markdown report file path",
    ),
    summary_path: Path = typer.Option(
        Path(".scratch/eda/aitod_eda_summary.json"),
        "--summary-path",
        help="Output JSON summary file path",
    ),
) -> None:
    """Run exploratory data analysis, generate scale diagnostics and report."""
    from src.data.eda import run_eda

    run_eda(
        data_dir=data_dir,
        figures_dir=figures_dir,
        report_path=report_path,
        summary_path=summary_path,
    )


@data_app.command("prepare")
def data_prepare(
    train_json: Path = typer.Option(
        Path("data/ai-tod-v2/annotations/aitodv2_train.json"),
        "--train-json",
        help="Path to training COCO annotations JSON",
    ),
    val_json: Path = typer.Option(
        Path("data/ai-tod-v2/annotations/aitod_val.json"),
        "--val-json",
        help="Path to validation COCO annotations JSON",
    ),
    source_train_imgs: Path = typer.Option(
        Path("data/ai-tod-v2/images/train"),
        "--source-train-imgs",
        help="Path to directory containing raw training images",
    ),
    source_val_imgs: Path = typer.Option(
        Path("data/ai-tod-v2/images/val"),
        "--source-val-imgs",
        help="Path to directory containing raw validation images",
    ),
    output_dir: Path = typer.Option(
        Path("data/ai-tod-v2/splits/mini"),
        "--output-dir",
        "-o",
        help="Destination directory for stratified mini split",
    ),
    train_size: int = typer.Option(
        1000,
        "--train-size",
        help="Number of training images for stratified mini split",
    ),
    val_size: int = typer.Option(
        200,
        "--val-size",
        help="Number of validation images for stratified mini split",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for deterministic sampling",
    ),
    config_yaml: Path = typer.Option(
        Path("configs/aitod_mini.yaml"),
        "--config-yaml",
        help="Path to write Ultralytics dataset configuration YAML",
    ),
    placeholders: bool = typer.Option(
        True,
        "--placeholders/--no-placeholders",
        help="Create placeholder images if source aerial image files are absent",
    ),
) -> None:
    """Generate stratified development mini split and convert annotations to YOLO format."""
    from src.data.converter import create_stratified_mini_dataset

    ui.print_header(
        "AI-TOD-v2 Stratified Development Subset & YOLO Converter",
        f"Output: {output_dir} | Train: {train_size} imgs | Val: {val_size} imgs",
    )

    if not train_json.exists():
        ui.print_error(f"Training annotation file not found: {train_json}")
        raise typer.Exit(code=1)

    if not val_json.exists():
        ui.print_error(f"Validation annotation file not found: {val_json}")
        raise typer.Exit(code=1)

    summary = create_stratified_mini_dataset(
        train_json=train_json,
        val_json=val_json,
        source_train_imgs=source_train_imgs,
        source_val_imgs=source_val_imgs,
        output_root=output_dir,
        train_size=train_size,
        val_size=val_size,
        seed=seed,
        config_yaml_path=config_yaml,
        create_placeholders_if_missing=placeholders,
    )

    train_conv = summary["train"]["conversion"]
    val_conv = summary["val"]["conversion"]

    ui.print_success(f"Generated Ultralytics dataset configuration at {config_yaml}")
    ui.print_table(
        "Conversion and Sanitization Summary",
        ["Metric", "Train Split", "Val Split"],
        [
            ["Images Processed", str(train_conv["images_processed"]), str(val_conv["images_processed"])],
            ["Labels Written", str(train_conv["labels_written"]), str(val_conv["labels_written"])],
            ["Total Anns Parsed", str(train_conv["total_annotations_seen"]), str(val_conv["total_annotations_seen"])],
            ["Valid Boxes Converted", str(train_conv["valid_converted_annotations"]), str(val_conv["valid_converted_annotations"])],
            ["Zero-dim Boxes Dropped", str(train_conv["zero_dim_annotations_dropped"]), str(val_conv["zero_dim_annotations_dropped"])],
            ["Out-of-bounds Dropped", str(train_conv["out_of_bounds_annotations_dropped"]), str(val_conv["out_of_bounds_annotations_dropped"])],
        ],
    )


if __name__ == "__main__":
    app()
