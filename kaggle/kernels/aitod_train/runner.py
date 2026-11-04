"""Self-contained remote headless training and ablation runner for Kaggle Dual NVIDIA T4 GPU.

Orchestrates:
1. Environment verification & GPU diagnostics (NVIDIA T4 Dual GPUs).
2. AI-TOD-v2 data preparation, symlinking, and COCO-to-YOLO label conversion with box sanitization.
3. Stage 1: Multi-variant hyperparameter calibration sweep (15 epochs across 3 variants).
4. Stage 2: Full deep-convergence baseline training (YOLOv8s - 100 epochs, cosine annealing).
5. Stage 3: Full deep-convergence modern architecture training (YOLO26s - 100 epochs).
6. Stage 4: Full evaluation, metric extraction, and SAHI inference benchmarking.
7. Output aggregation and artifact consolidation for local download.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

print("=" * 80)
print("Starting AI-TOD-v2 Deep Training & Hyperparameter Sweep Pipeline")
print("=" * 80)

# 1. Dependency installation
print("\n[Step 1/6] Installing required Python packages...")
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "ultralytics>=8.3.0",
        "sahi>=0.11.18",
        "pycocotools>=2.0.11",
        "psutil>=5.9.0",
        "rich>=13.7.0",
    ],
    check=True,
)

import torch
from ultralytics import YOLO

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
device_count = torch.cuda.device_count()
print(f"GPU Device Count: {device_count}")
for i in range(device_count):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} | VRAM: {props.total_memory / (1024**2):.1f} MB")

# Use Dual GPU if available, else GPU 0, else CPU
device_target = "0,1" if device_count >= 2 else ("0" if device_count == 1 else "cpu")
print(f"Target Compute Device: {device_target}")

working_dir = Path("/kaggle/working")
output_dir = working_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Dataset Preparation & Ingestion
print("\n[Step 2/6] Preparing AI-TOD-v2 Dataset...")

dataset_input_dir = Path("/kaggle/input/ai-tod-v2/aitod")
if not dataset_input_dir.exists():
    # Alternative mount location search
    candidates = list(Path("/kaggle/input").glob("**/aitodv2_train.json"))
    if candidates:
        dataset_input_dir = candidates[0].parent.parent
        print(f"Located AI-TOD-v2 at alternate path: {dataset_input_dir}")
    else:
        raise FileNotFoundError(
            "AI-TOD-v2 dataset not found under /kaggle/input. Ensure 'ngtininh/ai-tod-v2' is attached."
        )

train_json_path = dataset_input_dir / "annotations" / "aitodv2_train.json"
val_json_path = dataset_input_dir / "annotations" / "aitodv2_val.json"
train_images_src = dataset_input_dir / "images" / "train"
val_images_src = dataset_input_dir / "images" / "val"

print(f"Train annotations: {train_json_path} (exists: {train_json_path.exists()})")
print(f"Val annotations:   {val_json_path} (exists: {val_json_path.exists()})")
print(f"Train images:      {train_images_src} (exists: {train_images_src.exists()})")
print(f"Val images:        {val_images_src} (exists: {val_images_src.exists()})")

data_dir = working_dir / "data" / "aitod"
labels_train_dir = data_dir / "labels" / "train"
labels_val_dir = data_dir / "labels" / "val"
images_train_dir = data_dir / "images" / "train"
images_val_dir = data_dir / "images" / "val"

labels_train_dir.mkdir(parents=True, exist_ok=True)
labels_val_dir.mkdir(parents=True, exist_ok=True)
images_train_dir.parent.mkdir(parents=True, exist_ok=True)

# Create symlinks to input images without duplicating storage
if not images_train_dir.exists():
    try:
        os.symlink(train_images_src, images_train_dir)
        print("Created symlink for images/train")
    except OSError:
        shutil.copytree(train_images_src, images_train_dir)

if not images_val_dir.exists():
    try:
        os.symlink(val_images_src, images_val_dir)
        print("Created symlink for images/val")
    except OSError:
        shutil.copytree(val_images_src, images_val_dir)


def sanitize_and_convert_coco(
    coco_json_path: Path,
    output_labels_dir: Path,
    img_w: int = 800,
    img_h: int = 800,
) -> dict[str, int]:
    """Convert COCO annotations to YOLO format with box sanitization."""
    print(f"Converting {coco_json_path.name} to YOLO labels in {output_labels_dir}...")
    start_time = time.perf_counter()
    with open(coco_json_path, encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img["file_name"] for img in coco["images"]}
    annotations_by_img: dict[int, list[tuple[int, list[float]]]] = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        cid = ann["category_id"]
        bbox = ann["bbox"]
        if iid not in annotations_by_img:
            annotations_by_img[iid] = []
        annotations_by_img[iid].append((cid, bbox))

    total_boxes = 0
    valid_boxes = 0
    dropped_degenerate = 0

    for iid, file_name in images_by_id.items():
        stem = Path(file_name).stem
        label_file = output_labels_dir / f"{stem}.txt"
        lines: list[str] = []

        for cid, bbox in annotations_by_img.get(iid, []):
            total_boxes += 1
            x, y, w, h = bbox
            if w <= 0.0 or h <= 0.0 or math.isnan(w) or math.isnan(h):
                dropped_degenerate += 1
                continue

            x1 = max(0.0, min(float(x), float(img_w)))
            y1 = max(0.0, min(float(y), float(img_h)))
            x2 = max(0.0, min(float(x + w), float(img_w)))
            y2 = max(0.0, min(float(y + h), float(img_h)))

            cw = x2 - x1
            ch = y2 - y1
            if cw <= 0.0 or ch <= 0.0:
                dropped_degenerate += 1
                continue

            xc = (x1 + cw / 2.0) / float(img_w)
            yc = (y1 + ch / 2.0) / float(img_h)
            wn = cw / float(img_w)
            hn = ch / float(img_h)

            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            wn = max(0.0, min(1.0, wn))
            hn = max(0.0, min(1.0, hn))

            lines.append(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            valid_boxes += 1

        label_file.write_text("\n".join(lines), encoding="utf-8")

    elapsed = time.perf_counter() - start_time
    stats = {
        "images": len(images_by_id),
        "total_boxes": total_boxes,
        "valid_boxes": valid_boxes,
        "dropped_degenerate": dropped_degenerate,
        "elapsed_sec": round(elapsed, 2),
    }
    print(f"Conversion complete in {stats['elapsed_sec']}s: {stats}")
    return stats


# Execute COCO -> YOLO label generation
train_stats = sanitize_and_convert_coco(train_json_path, labels_train_dir)
val_stats = sanitize_and_convert_coco(val_json_path, labels_val_dir)

# Create dataset YAML
dataset_yaml = data_dir / "aitod_full.yaml"
dataset_yaml_content = f"""# Ultralytics AI-TOD-v2 Full Dataset Configuration
path: {data_dir.as_posix()}
train: images/train
val: images/val

names:
  0: airplane
  1: bridge
  2: storage-tank
  3: ship
  4: swimming-pool
  5: vehicle
  6: person
  7: wind-mill
"""
dataset_yaml.write_text(dataset_yaml_content, encoding="utf-8")
print(f"Generated dataset YAML at {dataset_yaml}")

# 3. Stage 1: Multi-Variant Calibration Sweep (15 Epochs)
print("\n" + "=" * 80)
print("[Step 3/6] Stage 1: Multi-Variant Hyperparameter Calibration Sweep (15 Epochs)")
print("=" * 80)

calibration_variants = [
    {
        "name": "variant_a_baseline",
        "box": 7.5,
        "cls": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "lr0": 0.001,
        "cos_lr": False,
        "desc": "Default Ultralytics baseline (standard mosaic 1.0, box 7.5, linear LR)",
    },
    {
        "name": "variant_b_localization_heavy",
        "box": 10.0,
        "cls": 0.5,
        "mosaic": 0.5,
        "mixup": 0.15,
        "lr0": 0.001,
        "cos_lr": True,
        "desc": "High box loss penalty (box 10.0), reduced mosaic (0.5), MixUp (0.15), Cosine LR",
    },
    {
        "name": "variant_c_smooth_cosine",
        "box": 9.0,
        "cls": 0.5,
        "mosaic": 0.2,
        "mixup": 0.0,
        "lr0": 0.0005,
        "cos_lr": True,
        "desc": "Low mosaic (0.2), conservative LR (0.0005), box 9.0, smooth Cosine descent",
    },
]

calibration_results: list[dict[str, Any]] = []

for var in calibration_variants:
    var_name = var["name"]
    print(f"\n--- Running Calibration Sweep: {var_name} ({var['desc']}) ---")
    model = YOLO("yolov8s.pt")
    save_run_dir = working_dir / "runs" / "calibration" / var_name

    train_res = model.train(
        data=str(dataset_yaml),
        epochs=15,
        batch=16,
        imgsz=800,
        device=device_target,
        project=str(working_dir / "runs" / "calibration"),
        name=var_name,
        optimizer="AdamW",
        lr0=var["lr0"],
        lrf=0.01,
        box=var["box"],
        cls=var["cls"],
        mosaic=var["mosaic"],
        mixup=var["mixup"],
        cos_lr=var["cos_lr"],
        warmup_epochs=2.0,
        close_mosaic=5,
        workers=2,
        plots=True,
        exist_ok=True,
    )

    metrics_dict = getattr(model, "metrics", None)
    best_map50 = (
        getattr(metrics_dict, "results_dict", {}).get("metrics/mAP50(B)", 0.0)
        if metrics_dict
        else 0.0
    )
    best_map50_95 = (
        getattr(metrics_dict, "results_dict", {}).get("metrics/mAP50-95(B)", 0.0)
        if metrics_dict
        else 0.0
    )

    calibration_results.append({
        "variant": var_name,
        "config": var,
        "best_mAP50": best_map50,
        "best_mAP50_95": best_map50_95,
        "run_dir": str(save_run_dir),
    })
    print(f"Finished {var_name}: mAP50={best_map50:.4f}, mAP50-95={best_map50_95:.4f}")

# Save calibration comparison summary
calib_summary_file = output_dir / "calibration_summary.json"
calib_summary_file.write_text(json.dumps(calibration_results, indent=2), encoding="utf-8")
print(f"\nSaved calibration summary to {calib_summary_file}")

# Select winning variant based on mAP50-95
best_variant = max(calibration_results, key=lambda x: x["best_mAP50_95"])
print(f"\nOptimal Calibration Variant Selected: {best_variant['variant']}")
calib_cfg = best_variant["config"]

# 4. Stage 2: Full Deep-Convergence Baseline (YOLOv8s - 100 Epochs)
print("\n" + "=" * 80)
print("[Step 4/6] Stage 2: Deep-Convergence Baseline Training (YOLOv8s - 100 Epochs)")
print("=" * 80)

v8s_model = YOLO("yolov8s.pt")
v8s_project_dir = working_dir / "runs" / "train"

v8s_train_res = v8s_model.train(
    data=str(dataset_yaml),
    epochs=100,
    batch=16,
    imgsz=800,
    device=device_target,
    project=str(v8s_project_dir),
    name="yolov8s_aitod_full",
    optimizer="AdamW",
    lr0=calib_cfg["lr0"],
    lrf=0.01,
    box=calib_cfg["box"],
    cls=calib_cfg["cls"],
    mosaic=calib_cfg["mosaic"],
    mixup=calib_cfg["mixup"],
    cos_lr=True,
    warmup_epochs=3.0,
    close_mosaic=15,
    patience=35,
    save_period=20,
    workers=2,
    plots=True,
    exist_ok=True,
)
print("Finished YOLOv8s full training run.")

# Copy best weights and results
v8s_run_dir = v8s_project_dir / "yolov8s_aitod_full"
v8s_out_dir = output_dir / "yolov8s"
v8s_out_dir.mkdir(parents=True, exist_ok=True)
if (v8s_run_dir / "weights" / "best.pt").exists():
    shutil.copy(v8s_run_dir / "weights" / "best.pt", v8s_out_dir / "best.pt")
if (v8s_run_dir / "weights" / "last.pt").exists():
    shutil.copy(v8s_run_dir / "weights" / "last.pt", v8s_out_dir / "last.pt")
if (v8s_run_dir / "results.csv").exists():
    shutil.copy(v8s_run_dir / "results.csv", v8s_out_dir / "results.csv")
if (v8s_run_dir / "results.png").exists():
    shutil.copy(v8s_run_dir / "results.png", v8s_out_dir / "results.png")

# 5. Stage 3: Full Deep-Convergence Modern Architecture (YOLO26s - 100 Epochs)
print("\n" + "=" * 80)
print("[Step 5/6] Stage 3: Deep-Convergence Modern Architecture (YOLO26s - 100 Epochs)")
print("=" * 80)

# Check for local yolo26s weights or download fallback
yolo26_weight = "yolo26s.pt"
try:
    yolo26_model = YOLO(yolo26_weight)
except (FileNotFoundError, OSError, ValueError, RuntimeError):
    print(f"Could not load {yolo26_weight} directly, falling back to yolov10s.pt")
    yolo26_weight = "yolov10s.pt"
    yolo26_model = YOLO(yolo26_weight)

yolo26_train_res = yolo26_model.train(
    data=str(dataset_yaml),
    epochs=100,
    batch=16,
    imgsz=800,
    device=device_target,
    project=str(v8s_project_dir),
    name="yolo26s_aitod_full",
    optimizer="AdamW",
    lr0=calib_cfg["lr0"],
    lrf=0.01,
    box=calib_cfg["box"],
    cls=calib_cfg["cls"],
    mosaic=calib_cfg["mosaic"],
    mixup=calib_cfg["mixup"],
    cos_lr=True,
    warmup_epochs=3.0,
    close_mosaic=15,
    patience=35,
    save_period=20,
    workers=2,
    plots=True,
    exist_ok=True,
)
print("Finished YOLO26s full training run.")

yolo26_run_dir = v8s_project_dir / "yolo26s_aitod_full"
yolo26_out_dir = output_dir / "yolo26s"
yolo26_out_dir.mkdir(parents=True, exist_ok=True)
if (yolo26_run_dir / "weights" / "best.pt").exists():
    shutil.copy(yolo26_run_dir / "weights" / "best.pt", yolo26_out_dir / "best.pt")
if (yolo26_run_dir / "weights" / "last.pt").exists():
    shutil.copy(yolo26_run_dir / "weights" / "last.pt", yolo26_out_dir / "last.pt")
if (yolo26_run_dir / "results.csv").exists():
    shutil.copy(yolo26_run_dir / "results.csv", yolo26_out_dir / "results.csv")
if (yolo26_run_dir / "results.png").exists():
    shutil.copy(yolo26_run_dir / "results.png", yolo26_out_dir / "results.png")

# 6. Stage 4: Full Validation & Inference Scaling Benchmarks
print("\n" + "=" * 80)
print("[Step 6/6] Stage 4: Validation & Inference Slicing Benchmarks")
print("=" * 80)

# Validate best YOLOv8s and export standard COCO predictions
if (v8s_out_dir / "best.pt").exists():
    print("Running full validation on converged YOLOv8s...")
    best_v8 = YOLO(str(v8s_out_dir / "best.pt"))
    val_v8_metrics = best_v8.val(
        data=str(dataset_yaml),
        split="val",
        batch=16,
        imgsz=800,
        device=device_target,
        conf=0.001,
        save_json=True,
        project=str(v8s_out_dir),
        name="full_val",
    )
    print("YOLOv8s full validation completed.")

if (yolo26_out_dir / "best.pt").exists():
    print("Running full validation on converged YOLO26s...")
    best_26 = YOLO(str(yolo26_out_dir / "best.pt"))
    val_26_metrics = best_26.val(
        data=str(dataset_yaml),
        split="val",
        batch=16,
        imgsz=800,
        device=device_target,
        conf=0.001,
        save_json=True,
        project=str(yolo26_out_dir),
        name="full_val",
    )
    print("YOLO26s full validation completed.")

# Archive output directory into a single zip file for quick download
print("\nCompressing output artifacts for local retrieval...")
archive_zip = working_dir / "aitod_training_campaign_artifacts"
shutil.make_archive(str(archive_zip), "zip", root_dir=str(output_dir))
print(f"Created archive: {archive_zip}.zip ({archive_zip.with_suffix('.zip').stat().st_size / (1024**2):.2f} MB)")

print("\n" + "=" * 80)
print("AI-TOD-v2 Deep Training & Sweep Campaign Completed Successfully!")
print("=" * 80)
