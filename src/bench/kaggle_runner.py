"""Headless remote training runner and orchestrator for Kaggle Dual NVIDIA T4 GPU sweeps."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi

from src.utils import ui

logger = logging.getLogger(__name__)

DEFAULT_KERNEL_ID = "miles2542/aitod-baseline-training"
DEFAULT_KERNEL_TITLE = "AI-TOD Baseline Training"


def get_authenticated_api() -> KaggleApi:
    """Initialize and authenticate Kaggle API client."""
    api = KaggleApi()
    api.authenticate()
    return api


def generate_kernel_scaffold(
    kernel_dir: Path,
    kernel_id: str = DEFAULT_KERNEL_ID,
    title: str = DEFAULT_KERNEL_TITLE,
    enable_gpu: bool = True,
    epochs: int = 10,
    batch_size: int = 8,
    img_size: int = 800,
) -> tuple[Path, Path]:
    """Generate self-contained headless Kaggle kernel configuration and runner script.

    Args:
        kernel_dir: Destination directory for kernel files.
        kernel_id: Kaggle kernel identifier (username/slug).
        title: Kernel display title on Kaggle.
        enable_gpu: Whether to activate Dual NVIDIA T4 GPU accelerator.
        epochs: Number of training epochs to execute in remote sweep.
        batch_size: Batch size for GPU dataloader.
        img_size: Input resolution in pixels.

    Returns:
        Tuple of (metadata_path, runner_script_path).
    """
    kernel_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = kernel_dir / "kernel-metadata.json"
    runner_path = kernel_dir / "runner.py"

    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": "runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if enable_gpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    runner_script = f'''"""Self-contained remote headless training runner for Kaggle Dual NVIDIA T4 GPU."""

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

print(f"PyTorch version: {{torch.__version__}}")
print(f"CUDA Available: {{torch.cuda.is_available()}}")
device_count = torch.cuda.device_count()
print(f"GPU Device Count: {{device_count}}")
for i in range(device_count):
    print(f"  Device {{i}}: {{torch.cuda.get_device_name(i)}}")

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
epochs = {epochs}
batch_size = {batch_size}
imgsz = {img_size}

for m_name in models_to_train:
    print(f"\\n================ Starting Baseline Training: {{m_name}} ================")
    model = YOLO(f"{{m_name}}.pt")
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
    print(f"Finished training {{m_name}}.")

print("\\nRemote training sweep finished successfully.")
'''
    runner_path.write_text(runner_script, encoding="utf-8")

    ui.print_success(f"Generated Kaggle kernel scaffold at {kernel_dir}")
    ui.print_info(f"Metadata: {metadata_path.name} | Script: {runner_path.name}")
    return metadata_path, runner_path


def push_kernel(kernel_dir: Path) -> dict[str, Any]:
    """Push kernel code and metadata to Kaggle to trigger remote execution."""
    metadata_path = kernel_dir / "kernel-metadata.json"
    if not metadata_path.exists():
        msg = f"kernel-metadata.json not found in {kernel_dir}"
        ui.print_error(msg)
        raise FileNotFoundError(msg)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    kernel_id = metadata.get("id", DEFAULT_KERNEL_ID)

    ui.print_header("Kaggle Remote Kernel Push", f"Kernel: {kernel_id} | Path: {kernel_dir}")
    api = get_authenticated_api()

    ui.print_info(f"Pushing kernel '{kernel_id}' to Kaggle...")
    response = api.kernels_push(str(kernel_dir))

    status_info: dict[str, Any] = {
        "kernel_id": kernel_id,
        "kernel_dir": str(kernel_dir),
        "status": getattr(response, "status", "pushed"),
        "error": getattr(response, "error", None),
        "url": getattr(response, "url", f"https://www.kaggle.com/code/{kernel_id}"),
    }

    if status_info.get("error"):
        ui.print_error(f"Push failed: {status_info['error']}")
    else:
        ui.print_success(f"Kernel pushed successfully. URL: {status_info['url']}")

    return status_info


def get_kernel_status(kernel_id: str = DEFAULT_KERNEL_ID) -> dict[str, Any]:
    """Check the real-time execution status of a Kaggle kernel."""
    ui.print_header("Kaggle Kernel Status Check", f"Kernel: {kernel_id}")
    api = get_authenticated_api()

    response = api.kernels_status(kernel_id)
    status_str = getattr(response, "status", "unknown")

    result = {
        "kernel_id": kernel_id,
        "status": status_str,
        "failure_message": getattr(response, "failure_message", None),
    }

    ui.print_table(
        "Kaggle Remote Execution Status",
        ["Attribute", "Value"],
        [
            ["Kernel ID", kernel_id],
            ["Execution Status", status_str],
            ["Failure Message", str(result["failure_message"] or "None")],
        ],
    )
    return result


def download_kernel_output(
    kernel_id: str = DEFAULT_KERNEL_ID,
    output_dir: Path = Path("runs/kaggle"),
) -> Path:
    """Download output artifacts, checkpoints, and logs from a completed Kaggle kernel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ui.print_header(
        "Download Kaggle Kernel Output",
        f"Kernel: {kernel_id} | Destination: {output_dir}",
    )
    api = get_authenticated_api()

    ui.print_info(f"Fetching remote outputs for '{kernel_id}' into {output_dir}...")
    api.kernels_output(kernel_id, path=str(output_dir))

    downloaded_files = list(output_dir.glob("**/*"))
    ui.print_success(f"Downloaded {len(downloaded_files)} files to {output_dir}")
    return output_dir
