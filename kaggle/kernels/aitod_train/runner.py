"""Headless empirical GPU training profiler for Kaggle Dual NVIDIA T4 GPU.

Profiles YOLOv8s across 6 epochs at 800x800 resolution with batch size 16 on NVIDIA T4.
Measures epoch latency, dataloader throughput, forward/backward execution time, and VRAM.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

print("=" * 70)
print("Initializing Kaggle Remote Execution Environment")
print("=" * 70)

# 1. Dependency installation
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "ultralytics>=8.3.0",
        "psutil>=5.9.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
    ],
    check=True,
)

import numpy as np
import psutil
import torch
from PIL import Image
from ultralytics import YOLO

print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
device_count = torch.cuda.device_count()
print(f"CUDA Device Count: {device_count}")
for i in range(device_count):
    props = torch.cuda.get_device_properties(i)
    print(
        f"  [GPU {i}] {props.name} | VRAM: {props.total_memory / (1024**2):.1f} MB | "
        f"Compute: {props.major}.{props.minor} | SMs: {props.multi_processor_count}"
    )

working_dir = Path("/kaggle/working")
runs_dir = working_dir / "runs"
runs_dir.mkdir(parents=True, exist_ok=True)

# 2. Dataset Synthesis: Generate self-contained AI-TOD-v2 mini split
print("\nPreparing AI-TOD-v2 mini-split on remote storage...")
data_root = working_dir / "data" / "ai-tod-v2" / "splits" / "mini"
train_imgs_dir = data_root / "images" / "train"
val_imgs_dir = data_root / "images" / "val"
train_lbls_dir = data_root / "labels" / "train"
val_lbls_dir = data_root / "labels" / "val"

train_imgs_dir.mkdir(parents=True, exist_ok=True)
val_imgs_dir.mkdir(parents=True, exist_ok=True)
train_lbls_dir.mkdir(parents=True, exist_ok=True)
val_lbls_dir.mkdir(parents=True, exist_ok=True)

template_path = data_root / "template.png"
if not template_path.exists():
    t_img = Image.new("RGB", (800, 800), color=(35, 40, 45))
    t_img.save(template_path, format="PNG")

import random
import shutil

random.seed(42)
NUM_TRAIN = 1000
NUM_VAL = 200

for i in range(NUM_TRAIN):
    img_name = f"train_{i:04d}.png"
    lbl_name = f"train_{i:04d}.txt"
    img_target = train_imgs_dir / img_name
    if not img_target.exists():
        try:
            os.link(template_path, img_target)
        except OSError:
            shutil.copyfile(template_path, img_target)

    num_boxes = random.randint(3, 8)
    lines = []
    for _ in range(num_boxes):
        cls_id = random.randint(0, 7)
        xc = round(random.uniform(0.05, 0.95), 6)
        yc = round(random.uniform(0.05, 0.95), 6)
        w = round(random.uniform(0.0075, 0.020), 6)
        h = round(random.uniform(0.0075, 0.020), 6)
        lines.append(f"{cls_id} {xc} {yc} {w} {h}\n")
    (train_lbls_dir / lbl_name).write_text("".join(lines), encoding="utf-8")

for i in range(NUM_VAL):
    img_name = f"val_{i:04d}.png"
    lbl_name = f"val_{i:04d}.txt"
    img_target = val_imgs_dir / img_name
    if not img_target.exists():
        try:
            os.link(template_path, img_target)
        except OSError:
            shutil.copyfile(template_path, img_target)

    num_boxes = random.randint(3, 6)
    lines = []
    for _ in range(num_boxes):
        cls_id = random.randint(0, 7)
        xc = round(random.uniform(0.05, 0.95), 6)
        yc = round(random.uniform(0.05, 0.95), 6)
        w = round(random.uniform(0.0075, 0.020), 6)
        h = round(random.uniform(0.0075, 0.020), 6)
        lines.append(f"{cls_id} {xc} {yc} {w} {h}\n")
    (val_lbls_dir / lbl_name).write_text("".join(lines), encoding="utf-8")

print(f"Dataset ready: {NUM_TRAIN} train images, {NUM_VAL} val images.")

config_yaml = working_dir / "aitod_mini_remote.yaml"
config_content = f"""path: {data_root.as_posix()}
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
config_yaml.write_text(config_content, encoding="utf-8")

# 3. Profiler Implementation
class GPUTrainingProfiler:
    def __init__(self, model_name: str, batch_size: int, imgsz: int) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.epoch_records: list[dict[str, Any]] = []
        self.batch_records: list[dict[str, Any]] = []

        self.current_epoch = 0
        self.epoch_start_time = 0.0
        self.prev_batch_end_time = 0.0
        self.batch_start_time = 0.0
        self.fwd_start_time = 0.0
        self.fwd_duration_ms = 0.0
        self.bwd_start_time = 0.0
        self.bwd_duration_ms = 0.0

        self.peak_vram_mb = 0.0
        self.peak_ram_mb = 0.0
        self.process = psutil.Process(os.getpid())

    def get_ram_mb(self) -> float:
        return self.process.memory_info().rss / (1024 * 1024)

    def get_vram_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0.0

    def attach_hooks(self, yolo_model: YOLO) -> None:
        torch_model = yolo_model.model

        def pre_forward_hook(module, inputs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.fwd_start_time = time.perf_counter()

        def post_forward_hook(module, inputs, outputs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.fwd_duration_ms = (time.perf_counter() - self.fwd_start_time) * 1000.0

        torch_model.register_forward_pre_hook(pre_forward_hook)
        torch_model.register_forward_hook(post_forward_hook)

        yolo_model.add_callback("on_train_epoch_start", self.on_epoch_start)
        yolo_model.add_callback("on_train_batch_start", self.on_batch_start)
        yolo_model.add_callback("optimizer_step", self.on_optimizer_step)
        yolo_model.add_callback("on_train_batch_end", self.on_batch_end)
        yolo_model.add_callback("on_train_epoch_end", self.on_epoch_end)

    def on_epoch_start(self, trainer: Any) -> None:
        self.current_epoch = getattr(trainer, "epoch", len(self.epoch_records)) + 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.epoch_start_time = time.perf_counter()
        self.prev_batch_end_time = self.epoch_start_time

    def on_batch_start(self, trainer: Any) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        dataloader_wait_ms = (now - self.prev_batch_end_time) * 1000.0 if self.prev_batch_end_time > 0 else 0.0
        self.batch_start_time = now
        self.current_dataloader_wait_ms = max(0.0, dataloader_wait_ms)
        self.bwd_start_time = time.perf_counter()

    def on_optimizer_step(self, trainer: Any) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.bwd_duration_ms = (time.perf_counter() - self.bwd_start_time) * 1000.0

    def on_batch_end(self, trainer: Any) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        total_batch_ms = (now - self.batch_start_time) * 1000.0
        self.prev_batch_end_time = now

        vram = self.get_vram_mb()
        ram = self.get_ram_mb()
        self.peak_vram_mb = max(self.peak_vram_mb, vram)
        self.peak_ram_mb = max(self.peak_ram_mb, ram)

        if self.bwd_duration_ms <= 0.0:
            self.bwd_duration_ms = max(0.0, total_batch_ms - self.fwd_duration_ms)

        self.batch_records.append({
            "epoch": self.current_epoch,
            "total_batch_ms": round(total_batch_ms, 2),
            "dataloader_wait_ms": round(self.current_dataloader_wait_ms, 2),
            "forward_ms": round(self.fwd_duration_ms, 2),
            "backward_ms": round(self.bwd_duration_ms, 2),
            "vram_mb": round(vram, 2),
            "ram_mb": round(ram, 2),
        })

    def on_epoch_end(self, trainer: Any) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_dur_sec = time.perf_counter() - self.epoch_start_time
        num_imgs = getattr(trainer, "num_batches", len(self.batch_records)) * self.batch_size
        if hasattr(trainer, "train_loader") and hasattr(trainer.train_loader, "dataset"):
            num_imgs = len(trainer.train_loader.dataset)

        fps = num_imgs / epoch_dur_sec if epoch_dur_sec > 0 else 0.0
        vram = self.get_vram_mb()
        ram = self.get_ram_mb()
        self.peak_vram_mb = max(self.peak_vram_mb, vram)
        self.peak_ram_mb = max(self.peak_ram_mb, ram)

        self.epoch_records.append({
            "epoch": self.current_epoch,
            "duration_sec": round(epoch_dur_sec, 2),
            "images": num_imgs,
            "throughput_fps": round(fps, 2),
            "vram_mb": round(vram, 2),
            "ram_mb": round(ram, 2),
        })


# 4. Run Training Benchmark on NVIDIA T4
target_device = 0 if device_count > 0 else "cpu"
batch_size = 16
epochs = 6
imgsz = 800
model_name = "yolov8s.pt"

print(f"\n================ Launching YOLOv8s Profiling ({epochs} Epochs, Batch {batch_size}, Device {target_device}) ================")
model = YOLO(model_name)
profiler = GPUTrainingProfiler(model_name=model_name, batch_size=batch_size, imgsz=imgsz)
profiler.attach_hooks(model)

t_benchmark_start = time.perf_counter()
model.train(
    data=str(config_yaml),
    epochs=epochs,
    batch=batch_size,
    imgsz=imgsz,
    device=target_device,
    workers=2,
    val=False,
    plots=False,
    save=True,
    project=str(runs_dir),
    name="kaggle_gpu_benchmark",
    exist_ok=True,
)
total_wall_clock_sec = time.perf_counter() - t_benchmark_start

epoch_durations = [r["duration_sec"] for r in profiler.epoch_records]
fps_list = [r["throughput_fps"] for r in profiler.epoch_records]
fwd_times = [b["forward_ms"] for b in profiler.batch_records if b["forward_ms"] > 0]
bwd_times = [b["backward_ms"] for b in profiler.batch_records if b["backward_ms"] > 0]
dl_times = [b["dataloader_wait_ms"] for b in profiler.batch_records]

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Fallback"

summary = {
    "environment": {
        "platform": f"{platform.system()} {platform.release()}",
        "device_count": device_count,
        "primary_gpu": gpu_name,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "pytorch_version": torch.__version__,
    },
    "config": {
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "dataset_images": NUM_TRAIN,
        "total_wall_clock_sec": round(total_wall_clock_sec, 2),
    },
    "epoch_metrics": {
        "epochs_run": len(profiler.epoch_records),
        "mean_epoch_sec": round(float(np.mean(epoch_durations)), 2) if epoch_durations else 0.0,
        "std_epoch_sec": round(float(np.std(epoch_durations)), 2) if epoch_durations else 0.0,
        "min_epoch_sec": round(float(np.min(epoch_durations)), 2) if epoch_durations else 0.0,
        "max_epoch_sec": round(float(np.max(epoch_durations)), 2) if epoch_durations else 0.0,
        "epoch_details": profiler.epoch_records,
    },
    "throughput_metrics": {
        "mean_throughput_fps": round(float(np.mean(fps_list)), 2) if fps_list else 0.0,
        "std_throughput_fps": round(float(np.std(fps_list)), 2) if fps_list else 0.0,
        "min_throughput_fps": round(float(np.min(fps_list)), 2) if fps_list else 0.0,
        "max_throughput_fps": round(float(np.max(fps_list)), 2) if fps_list else 0.0,
    },
    "step_breakdown_ms": {
        "mean_dataloader_wait_ms": round(float(np.mean(dl_times)), 2) if dl_times else 0.0,
        "mean_forward_ms": round(float(np.mean(fwd_times)), 2) if fwd_times else 0.0,
        "mean_backward_ms": round(float(np.mean(bwd_times)), 2) if bwd_times else 0.0,
        "mean_total_batch_ms": round(float(np.mean([b["total_batch_ms"] for b in profiler.batch_records])), 2) if profiler.batch_records else 0.0,
    },
    "memory_mb": {
        "peak_vram_mb": round(profiler.peak_vram_mb, 2),
        "peak_ram_mb": round(profiler.peak_ram_mb, 2),
    },
}

summary_json_path = runs_dir / "kaggle_gpu_benchmark.json"
summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

# Write CSV summary
csv_path = runs_dir / "summary.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("metric,value\n")
    f.write(f"primary_gpu,{gpu_name}\n")
    f.write(f"epochs_run,{summary['epoch_metrics']['epochs_run']}\n")
    f.write(f"mean_epoch_sec,{summary['epoch_metrics']['mean_epoch_sec']}\n")
    f.write(f"std_epoch_sec,{summary['epoch_metrics']['std_epoch_sec']}\n")
    f.write(f"mean_throughput_fps,{summary['throughput_metrics']['mean_throughput_fps']}\n")
    f.write(f"mean_forward_ms,{summary['step_breakdown_ms']['mean_forward_ms']}\n")
    f.write(f"mean_backward_ms,{summary['step_breakdown_ms']['mean_backward_ms']}\n")
    f.write(f"mean_dataloader_wait_ms,{summary['step_breakdown_ms']['mean_dataloader_wait_ms']}\n")
    f.write(f"peak_vram_mb,{summary['memory_mb']['peak_vram_mb']}\n")
    f.write(f"peak_ram_mb,{summary['memory_mb']['peak_ram_mb']}\n")
    f.write(f"total_wall_clock_sec,{summary['config']['total_wall_clock_sec']}\n")

print("\n" + "=" * 70)
print("KAGGLE GPU BENCHMARK COMPLETED")
print("=" * 70)
print(f"Primary GPU: {gpu_name}")
print(f"Mean Epoch Duration: {summary['epoch_metrics']['mean_epoch_sec']} ± {summary['epoch_metrics']['std_epoch_sec']} s")
print(f"Mean Throughput: {summary['throughput_metrics']['mean_throughput_fps']} imgs/sec")
print(f"Mean Forward Step: {summary['step_breakdown_ms']['mean_forward_ms']} ms/batch")
print(f"Mean Backward Step: {summary['step_breakdown_ms']['mean_backward_ms']} ms/batch")
print(f"Dataloader Wait: {summary['step_breakdown_ms']['mean_dataloader_wait_ms']} ms/batch")
print(f"Peak VRAM Footprint: {summary['memory_mb']['peak_vram_mb']} MB")
print(f"Total Wall-Clock: {summary['config']['total_wall_clock_sec']} s")
print("=" * 70)
