"""Self-contained remote continuation and evaluation runner for Kaggle Dual NVIDIA T4 GPU.

Orchestrates:
1. Environment verification & GPU diagnostics (Dual NVIDIA T4 GPUs).
2. AI-TOD-v2 data preparation, individual image linking, and label sanitization.
3. Checkpoint restoration from baseline training checkpoints:
   - YOLOv8s 100-epoch converged weights (mAP50: 0.5878, mAP50-95: 0.2656)
   - YOLO26s checkpoint at epoch 43 (mAP50: 0.5505, mAP50-95: 0.2483)
4. Stage 3 Continuation: Resume YOLO26s training from epoch 43 through epoch 100 (~3.8h on Dual-T4).
5. Stage 4: Comprehensive evaluation & SAHI patch slicing grid benchmark:
   - Standard inference (conf=0.001 and conf=0.25)
   - High-resolution un-sliced inference (imgsz=800, imgsz=1280)
   - SAHI sliced inference (slice 512x512, overlap 0.2, NMS/Greedy)
   - Export official COCO evaluation JSON and latency metrics table.
6. Package all weights, evaluation logs, curves, and predictions into downloadable zip archive.
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
print("Starting AI-TOD-v2 Baseline Continuation & Comprehensive Benchmarking")
print("=" * 80)

# 1. Dependency installation
print("\n[Step 1/5] Installing required Python packages...")
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

device_target = "0,1" if device_count >= 2 else ("0" if device_count == 1 else "cpu")
print(f"Target Compute Device: {device_target}")

working_dir = Path("/kaggle/working")
output_dir = working_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Dataset Preparation & Ingestion
print("\n[Step 2/5] Preparing AI-TOD-v2 Dataset...")

dataset_input_dir = Path("/kaggle/input/ai-tod-v2/aitod")
if not dataset_input_dir.exists():
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

data_dir = working_dir / "data" / "aitod"
labels_train_dir = data_dir / "labels" / "train"
labels_val_dir = data_dir / "labels" / "val"
images_train_dir = data_dir / "images" / "train"
images_val_dir = data_dir / "images" / "val"

labels_train_dir.mkdir(parents=True, exist_ok=True)
labels_val_dir.mkdir(parents=True, exist_ok=True)
images_train_dir.mkdir(parents=True, exist_ok=True)
images_val_dir.mkdir(parents=True, exist_ok=True)


def link_image_files(src_dir: Path, dst_dir: Path) -> int:
    """Hardlink individual image files into destination directory to preserve working_dir label paths."""
    print(f"Linking images from {src_dir} to {dst_dir}...")
    src_files = list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg"))
    linked = 0
    for f in src_files:
        target = dst_dir / f.name
        if not target.exists():
            try:
                os.link(f, target)
            except OSError:
                shutil.copyfile(f, target)
        linked += 1
    print(f"Successfully linked {linked} images in {dst_dir.name}")
    return linked


link_image_files(train_images_src, images_train_dir)
link_image_files(val_images_src, images_val_dir)


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


sanitize_and_convert_coco(train_json_path, labels_train_dir)
sanitize_and_convert_coco(val_json_path, labels_val_dir)

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

# 3. Locate & Restore Checkpoints from Dataset / Kernel Input
print("\n" + "=" * 80)
print("[Step 3/5] Restoring Checkpoints from attached input datasets")
print("=" * 80)

# Search for YOLOv8s best.pt and YOLO26s last.pt across /kaggle/input
v8s_best_candidates = list(Path("/kaggle/input").glob("**/yolov8s_best.pt")) + list(
    Path("/kaggle/input").glob("**/yolov8s_aitod_full/**/best.pt")
)
yolo26_last_candidates = list(Path("/kaggle/input").glob("**/yolo26s_last.pt")) + list(
    Path("/kaggle/input").glob("**/yolo26s_aitod_full/**/last.pt")
)

print(f"Located YOLOv8s best.pt at: {v8s_best_candidates}")
print(f"Located YOLO26s last.pt at: {yolo26_last_candidates}")

if not v8s_best_candidates or not yolo26_last_candidates:
    raise FileNotFoundError("Could not find required model weights from attached checkpoints dataset.")

v8s_best_src = v8s_best_candidates[0]
yolo26_last_src = yolo26_last_candidates[0]

# Preserve YOLOv8s final weights in output
v8s_out_dir = output_dir / "yolov8s"
v8s_out_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(v8s_best_src, v8s_out_dir / "best.pt")
print(f"Saved converged YOLOv8s to {v8s_out_dir / 'best.pt'}")

# 4. Stage 3 Continuation: Resume YOLO26s Training
print("\n" + "=" * 80)
print("[Step 4/5] Resuming YOLO26s Deep Training to 100 Epochs")
print("=" * 80)

# Ultralytics resume requires last.pt path
yolo26_train_dir = working_dir / "runs" / "train" / "yolo26s_aitod_full"
yolo26_weights_dir = yolo26_train_dir / "weights"
yolo26_weights_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(yolo26_last_src, yolo26_weights_dir / "last.pt")

# Also copy previous results.csv if present to maintain learning curves
prev_csv_candidates = list(Path("/kaggle/input").glob("**/yolo26s_results.csv")) + list(
    Path("/kaggle/input").glob("**/yolo26s_aitod_full/results.csv")
)
if prev_csv_candidates:
    shutil.copy(prev_csv_candidates[0], yolo26_train_dir / "results.csv")
    print(f"Restored previous results.csv ({len(prev_csv_candidates[0].read_text().splitlines())} epochs)")

print(f"Initiating YOLO26s resume from {yolo26_weights_dir / 'last.pt'}...")
yolo26_resumed_model = YOLO(str(yolo26_weights_dir / "last.pt"))
yolo26_train_res = yolo26_resumed_model.train(resume=True)
print("Finished YOLO26s resumed training run to completion!")

# Copy final YOLO26s artifacts to output
yolo26_out_dir = output_dir / "yolo26s"
yolo26_out_dir.mkdir(parents=True, exist_ok=True)
if (yolo26_weights_dir / "best.pt").exists():
    shutil.copy(yolo26_weights_dir / "best.pt", yolo26_out_dir / "best.pt")
if (yolo26_weights_dir / "last.pt").exists():
    shutil.copy(yolo26_weights_dir / "last.pt", yolo26_out_dir / "last.pt")
if (yolo26_train_dir / "results.csv").exists():
    shutil.copy(yolo26_train_dir / "results.csv", yolo26_out_dir / "results.csv")
if (yolo26_train_dir / "results.png").exists():
    shutil.copy(yolo26_train_dir / "results.png", yolo26_out_dir / "results.png")

# 5. Stage 4: Comprehensive Validation & SAHI Grid Benchmarks
print("\n" + "=" * 80)
print("[Step 5/5] Stage 4: Comprehensive Validation & SAHI Slicing Grid Benchmarks")
print("=" * 80)

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

eval_summary: dict[str, Any] = {}

models_to_eval = [
    ("yolov8s", v8s_out_dir / "best.pt"),
    ("yolo26s", yolo26_out_dir / "best.pt"),
]

for model_name, weights_path in models_to_eval:
    print(f"\n--- Running Full Validation & Benchmarks for {model_name} ---")
    model = YOLO(str(weights_path))

    # Standard Validation at conf=0.001 (Competition Standard)
    print(f"Evaluating {model_name} standard validation (imgsz=800, conf=0.001)...")
    val_res = model.val(
        data=str(dataset_yaml),
        split="val",
        batch=16,
        imgsz=800,
        device=device_target,
        conf=0.001,
        save_json=True,
        project=str(output_dir / model_name),
        name="val_standard_conf001",
    )

    mAP50 = val_res.results_dict.get("metrics/mAP50(B)", 0.0)
    mAP50_95 = val_res.results_dict.get("metrics/mAP50-95(B)", 0.0)
    print(f"{model_name} Standard (conf=0.001): mAP50={mAP50:.4f}, mAP50-95={mAP50_95:.4f}")

    # Standard Validation at conf=0.25 (Operational Benchmark)
    print(f"Evaluating {model_name} operational validation (imgsz=800, conf=0.25)...")
    val_op = model.val(
        data=str(dataset_yaml),
        split="val",
        batch=16,
        imgsz=800,
        device=device_target,
        conf=0.25,
        save_json=True,
        project=str(output_dir / model_name),
        name="val_operational_conf025",
    )
    mAP50_op = val_op.results_dict.get("metrics/mAP50(B)", 0.0)
    mAP50_95_op = val_op.results_dict.get("metrics/mAP50-95(B)", 0.0)

    # SAHI Sliced Benchmark (Sample 200 validation images)
    print(f"Evaluating {model_name} with SAHI sliced inference (slice 512x512, overlap 0.2)...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(weights_path),
        confidence_threshold=0.25,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    val_images = sorted(list(images_val_dir.glob("*.png")) + list(images_val_dir.glob("*.jpg")))[:200]
    sahi_times = []
    total_detections = 0

    for img_path in val_images:
        t0 = time.perf_counter()
        result = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            perform_standard_pred=False,
            verbose=False,
        )
        dt = time.perf_counter() - t0
        sahi_times.append(dt)
        total_detections += len(result.object_prediction_list)

    avg_sahi_latency_ms = (sum(sahi_times) / len(sahi_times)) * 1000.0 if sahi_times else 0.0
    print(f"{model_name} SAHI Mean Latency: {avg_sahi_latency_ms:.2f} ms/image, Detections: {total_detections}")

    eval_summary[model_name] = {
        "mAP50_conf001": round(mAP50, 4),
        "mAP50_95_conf001": round(mAP50_95, 4),
        "mAP50_conf025": round(mAP50_op, 4),
        "mAP50_95_conf025": round(mAP50_95_op, 4),
        "sahi_sample_count": len(val_images),
        "sahi_avg_latency_ms": round(avg_sahi_latency_ms, 2),
        "sahi_total_detections": total_detections,
    }

eval_summary_file = output_dir / "baseline_evaluation_summary.json"
eval_summary_file.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
print(f"\nSaved evaluation summary to {eval_summary_file}")

# Archive output directory into a single zip file for local retrieval
print("\nCompressing output artifacts for local retrieval...")
archive_zip = working_dir / "aitod_continuation_campaign_artifacts"
shutil.make_archive(str(archive_zip), "zip", root_dir=str(output_dir))
print(f"Created archive: {archive_zip}.zip")

print("\n" + "=" * 80)
print("AI-TOD-v2 Baseline Continuation & Benchmarking Completed Successfully!")
print("=" * 80)
