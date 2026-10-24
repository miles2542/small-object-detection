"""Self-contained remote headless training runner for Kaggle Dual NVIDIA T4 GPU."""

import os
import subprocess
import sys
from pathlib import Path

# 1. Install prerequisites in remote Kaggle kernel
print("Installing required packages...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "ultralytics>=8.3.0", "sahi>=0.11.18", "gdown>=6.4.2", "rich>=13.7.0"
], check=True)

import torch
from ultralytics import YOLO

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
device_count = torch.cuda.device_count()
print(f"GPU Device Count: {device_count}")
for i in range(device_count):
    print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

# Configure target device
target_device = "0" if device_count > 0 else "cpu"

working_dir = Path("/kaggle/working")
data_dir = working_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# 2. Fetch AI-TOD-v2 annotations and data split
import gdown

print("Fetching dataset annotations...")
ann_zip = working_dir / "annotations_v2.zip"
gdown.download(id="131jpoF9oThL2L8y0CU0u9JI1vy76WrXC", output=str(ann_zip), quiet=False)

import zipfile
with zipfile.ZipFile(ann_zip, "r") as zf:
    zf.extractall(data_dir / "ai-tod-v2")
if ann_zip.exists():
    ann_zip.unlink()

# 3. Create dataset YAML configuration for remote execution
config_yaml = working_dir / "aitod_remote.yaml"
config_content = """# Remote Kaggle AI-TOD configuration
path: /kaggle/working/data/ai-tod-v2/splits/mini
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

# 4. Execute training runs across baselines
models_to_train = ["yolov8s", "yolo26s"]
epochs = 10
batch_size = 8
imgsz = 800

for m_name in models_to_train:
    print(f"\n================ Starting Baseline Training: {m_name} ================")
    model = YOLO(f"{m_name}.pt")
    results = model.train(
        data=str(config_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=target_device,
        project=str(working_dir / "runs" / "train"),
        name=m_name,
        workers=2,
        exist_ok=True,
    )
    print(f"Finished training {m_name}.")

print("\nRemote training sweep finished successfully.")
