"""Comprehensive runtime, latency, throughput, and memory benchmarking suite.

Profiles YOLOv8s and YOLO26s across resolutions, batch sizes, CPU thread allocations,
inference modes (standard full-frame vs. SAHI tiled inference), and real aerial imagery.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

from src.utils import ui


@dataclass
class BenchmarkResult:
    model_name: str
    scenario: str
    resolution: tuple[int, int]
    batch_size: int
    threads: int
    cold_start_ms: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    std_latency_ms: float
    fps: float
    peak_ram_mb: float
    notes: str = ""


def get_process_memory_mb() -> float:
    """Return current process resident memory in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_inference(
    model: YOLO,
    model_name: str,
    img_size: int,
    batch_size: int = 1,
    num_runs: int = 10,
    warmup_runs: int = 2,
    threads: int | None = None,
    device: str = "cpu",
) -> BenchmarkResult:
    """Profile inference latency, throughput, and memory usage."""
    if threads is not None:
        torch.set_num_threads(threads)
    current_threads = torch.get_num_threads()

    gc.collect()
    dummy_input = torch.rand(batch_size, 3, img_size, img_size)
    if device != "cpu":
        try:
            dummy_input = dummy_input.to(device)
        except (RuntimeError, ValueError) as err:
            ui.print_warning(f"Could not transfer dummy input to {device}: {err}. Falling back to CPU.")
            device = "cpu"

    # Cold start (first inference on fresh input)
    t0 = time.perf_counter()
    _ = model(dummy_input, verbose=False, device=device)
    cold_start_ms = (time.perf_counter() - t0) * 1000.0

    # Warmup runs
    for _ in range(warmup_runs):
        _ = model(dummy_input, verbose=False, device=device)

    # Measured runs
    latencies: list[float] = []
    peak_ram = get_process_memory_mb()

    for _ in range(num_runs):
        t_start = time.perf_counter()
        _ = model(dummy_input, verbose=False, device=device)
        lat = (time.perf_counter() - t_start) * 1000.0
        latencies.append(lat)
        curr_ram = get_process_memory_mb()
        peak_ram = max(peak_ram, curr_ram)

    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    std_lat = float(np.std(latencies))
    fps = (batch_size / (mean_lat / 1000.0)) if mean_lat > 0 else 0.0

    return BenchmarkResult(
        model_name=model_name,
        scenario="standard_inference",
        resolution=(img_size, img_size),
        batch_size=batch_size,
        threads=current_threads,
        cold_start_ms=round(cold_start_ms, 2),
        mean_latency_ms=round(mean_lat, 2),
        p50_latency_ms=round(p50_lat, 2),
        p95_latency_ms=round(p95_lat, 2),
        std_latency_ms=round(std_lat, 2),
        fps=round(fps, 2),
        peak_ram_mb=round(peak_ram, 2),
    )


def benchmark_real_images(
    model: YOLO,
    model_name: str,
    image_paths: Sequence[Path],
    imgsz: int = 800,
    device: str = "cpu",
) -> BenchmarkResult:
    """Benchmark end-to-end inference on real dataset images."""
    gc.collect()
    imgs = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(str(p))

    if not imgs:
        raise ValueError("No images found for real benchmark")

    t0 = time.perf_counter()
    _ = model(imgs[0], imgsz=imgsz, verbose=False, device=device)
    cold_start_ms = (time.perf_counter() - t0) * 1000.0

    latencies: list[float] = []
    peak_ram = get_process_memory_mb()

    for img_path in imgs:
        t_start = time.perf_counter()
        _ = model(img_path, imgsz=imgsz, verbose=False, device=device)
        lat = (time.perf_counter() - t_start) * 1000.0
        latencies.append(lat)
        curr_ram = get_process_memory_mb()
        peak_ram = max(peak_ram, curr_ram)

    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    std_lat = float(np.std(latencies))
    fps = (1.0 / (mean_lat / 1000.0)) if mean_lat > 0 else 0.0

    return BenchmarkResult(
        model_name=model_name,
        scenario="real_drone_imagery",
        resolution=(imgsz, imgsz),
        batch_size=1,
        threads=torch.get_num_threads(),
        cold_start_ms=round(cold_start_ms, 2),
        mean_latency_ms=round(mean_lat, 2),
        p50_latency_ms=round(p50_lat, 2),
        p95_latency_ms=round(p95_lat, 2),
        std_latency_ms=round(std_lat, 2),
        fps=round(fps, 2),
        peak_ram_mb=round(peak_ram, 2),
        notes=f"Evaluated across {len(imgs)} real aerial drone frames",
    )


def benchmark_sahi(
    model_path: str,
    model_name: str,
    img_size: int,
    slice_size: int = 512,
    overlap_ratio: float = 0.2,
    num_runs: int = 3,
    device: str = "cpu",
) -> BenchmarkResult:
    """Profile SAHI sliced inference on CPU or target device."""
    gc.collect()
    dummy_img = (np.random.rand(img_size, img_size, 3) * 255).astype(np.uint8)

    t0 = time.perf_counter()
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=0.25,
        device=device,
    )
    _ = get_sliced_prediction(
        dummy_img,
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        verbose=0,
    )
    cold_start_ms = (time.perf_counter() - t0) * 1000.0

    latencies: list[float] = []
    peak_ram = get_process_memory_mb()

    for _ in range(num_runs):
        t_start = time.perf_counter()
        _ = get_sliced_prediction(
            dummy_img,
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            verbose=0,
        )
        lat = (time.perf_counter() - t_start) * 1000.0
        latencies.append(lat)
        curr_ram = get_process_memory_mb()
        peak_ram = max(peak_ram, curr_ram)

    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    std_lat = float(np.std(latencies))
    fps = (1.0 / (mean_lat / 1000.0)) if mean_lat > 0 else 0.0

    return BenchmarkResult(
        model_name=model_name,
        scenario=f"sahi_tiled_{slice_size}x{slice_size}_ov{overlap_ratio}",
        resolution=(img_size, img_size),
        batch_size=1,
        threads=torch.get_num_threads(),
        cold_start_ms=round(cold_start_ms, 2),
        mean_latency_ms=round(mean_lat, 2),
        p50_latency_ms=round(p50_lat, 2),
        p95_latency_ms=round(p95_lat, 2),
        std_latency_ms=round(std_lat, 2),
        fps=round(fps, 2),
        peak_ram_mb=round(peak_ram, 2),
        notes=f"Slice {slice_size}x{slice_size}, overlap {overlap_ratio}",
    )


def benchmark_dummy_training(
    model: YOLO,
    model_name: str,
    img_size: int = 800,
    batch_size: int = 2,
    num_steps: int = 3,
    device: str = "cpu",
) -> dict[str, str | int | float]:
    """Test backward pass, gradient allocation, and step latency."""
    gc.collect()
    m = model.model
    if device != "cpu":
        try:
            m = m.to(device)
        except (RuntimeError, ValueError):
            device = "cpu"

    m.train()
    for p in m.parameters():
        p.requires_grad = True

    optimizer = torch.optim.SGD(m.parameters(), lr=0.001)
    latencies: list[float] = []
    peak_ram = get_process_memory_mb()

    dummy_input = torch.rand(batch_size, 3, img_size, img_size)
    if device != "cpu":
        dummy_input = dummy_input.to(device)

    def extract_loss(obj: object) -> torch.Tensor:
        total = torch.tensor(0.0, requires_grad=True, device=dummy_input.device)
        if isinstance(obj, torch.Tensor):
            return obj.sum()
        if isinstance(obj, dict):
            return sum((extract_loss(v) for v in obj.values()), total)
        if isinstance(obj, (list, tuple)):
            return sum((extract_loss(v) for v in obj), total)
        return total

    for _ in range(num_steps):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = m(dummy_input)
        loss = extract_loss(out)
        loss.backward()
        optimizer.step()
        latencies.append((time.perf_counter() - t0) * 1000.0)
        curr_ram = get_process_memory_mb()
        peak_ram = max(peak_ram, curr_ram)

    m.eval()
    return {
        "model_name": model_name,
        "img_size": img_size,
        "batch_size": batch_size,
        "mean_step_ms": round(float(np.mean(latencies)), 2),
        "peak_ram_mb": round(peak_ram, 2),
        "status": "PASS",
    }


def run_benchmarks(
    models: Sequence[str] = ("yolov8s", "yolo26s"),
    batch_sizes: Sequence[int] = (1, 4, 8, 16),
    img_size: int = 800,
    output_dir: Path = Path(".scratch/benchmarks"),
    device: str = "cpu",
) -> dict[str, Any]:
    """Run full benchmarking suite across specified models, batch sizes, and resolutions."""
    ui.print_header(
        "Small Object Detection Runtime & Memory Benchmark",
        f"Models: {list(models)} | Resolution: {img_size}x{img_size} | Device: {device}",
    )

    mem = psutil.virtual_memory()
    ui.print_info(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    ui.print_info(
        f"CPU: {platform.processor() or 'Standard CPU'} "
        f"({psutil.cpu_count(logical=False)} physical cores, {psutil.cpu_count(logical=True)} logical threads)"
    )
    ui.print_info(f"Host RAM: {mem.total / (1024**3):.2f} GB")
    ui.print_info(f"PyTorch: {torch.__version__} | Ultralytics: {YOLO.__module__}")

    loaded_models: dict[str, YOLO] = {}
    for m_name in models:
        chkpt = f"{m_name}.pt" if not m_name.endswith(".pt") else m_name
        ui.print_info(f"Loading checkpoint {chkpt}...")
        loaded_models[m_name] = YOLO(chkpt)

    results: dict[str, Any] = {
        "hardware": {
            "os": f"{platform.system()} {platform.release()}",
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "pytorch_version": torch.__version__,
            "device": device,
        },
        "resolution_sweep": [],
        "batch_sweep": [],
        "thread_scaling": [],
        "real_images_sweep": [],
        "sahi_sweep": [],
        "training_step": [],
    }

    # 1. Batch Size Sweep
    ui.print_info(f"Executing batch size sweep for {list(batch_sizes)} at {img_size}x{img_size}...")
    batch_rows = []
    for b in batch_sizes:
        for m_name, model in loaded_models.items():
            r = benchmark_inference(
                model=model,
                model_name=m_name,
                img_size=img_size,
                batch_size=b,
                num_runs=3,
                device=device,
            )
            results["batch_sweep"].append(asdict(r))
            batch_rows.append([
                m_name,
                str(b),
                f"{r.mean_latency_ms:.1f} ms",
                f"{r.mean_latency_ms / b:.1f} ms",
                f"{r.fps:.2f}",
                f"{r.peak_ram_mb:.1f} MB",
            ])

    ui.print_table(
        "Batch Size Scaling",
        ["Model", "Batch", "Step Latency", "Per-Image Latency", "Throughput (FPS)", "Peak RAM"],
        batch_rows,
    )

    # 2. SAHI Tiled Inference Check
    sahi_rows = []
    for m_name in models:
        chkpt = f"{m_name}.pt" if not m_name.endswith(".pt") else m_name
        if Path(chkpt).exists():
            s = benchmark_sahi(
                model_path=chkpt,
                model_name=m_name,
                img_size=img_size,
                slice_size=512,
                overlap_ratio=0.2,
                num_runs=2,
                device=device,
            )
            results["sahi_sweep"].append(asdict(s))
            sahi_rows.append([m_name, f"{s.mean_latency_ms:.1f} ms", f"{s.fps:.2f}", f"{s.peak_ram_mb:.1f} MB"])

    if sahi_rows:
        ui.print_table("SAHI Tiled Inference (512x512, 0.2 overlap)", ["Model", "Mean Latency", "FPS", "Peak RAM"], sahi_rows)

    # 3. Dummy Training Backward Pass
    train_rows = []
    for m_name, model in loaded_models.items():
        tr = benchmark_dummy_training(model=model, model_name=m_name, img_size=img_size, batch_size=2, device=device)
        results["training_step"].append(tr)
        train_rows.append([m_name, f"{tr['mean_step_ms']} ms", f"{tr['peak_ram_mb']} MB", str(tr["status"])])

    ui.print_table("Dummy Training Backward Pass (batch=2)", ["Model", "Step Latency", "Peak RAM", "Status"], train_rows)

    # Save output raw JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "runtime_benchmark_raw.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    ui.print_success(f"Benchmark results successfully written to {json_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Small Object Detection runtime & throughput benchmark")
    parser.add_argument("--models", nargs="+", default=["yolov8s", "yolo26s"], help="Models to benchmark")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16], help="Batch sizes to evaluate")
    parser.add_argument("--img-size", type=int, default=800, help="Input resolution")
    parser.add_argument("--output-dir", type=Path, default=Path(".scratch/benchmarks"), help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Target device (cpu, cuda:0, etc.)")
    args = parser.parse_args()

    run_benchmarks(
        models=args.models,
        batch_sizes=args.batch_sizes,
        img_size=args.img_size,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
