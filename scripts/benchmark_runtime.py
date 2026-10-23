"""Comprehensive runtime, latency, throughput, and memory benchmarking suite.

Profiles YOLOv8s and YOLO26s across resolutions, batch sizes, CPU thread allocations,
inference modes (standard full-frame vs. SAHI tiled inference), and real drone imagery.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO


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
) -> BenchmarkResult:
    """Profile inference latency, throughput, and memory usage."""
    if threads is not None:
        torch.set_num_threads(threads)
    current_threads = torch.get_num_threads()

    gc.collect()
    dummy_input = torch.rand(batch_size, 3, img_size, img_size)

    # Cold start (first inference on fresh input)
    t0 = time.perf_counter()
    _ = model(dummy_input, verbose=False)
    cold_start_ms = (time.perf_counter() - t0) * 1000.0

    # Warmup runs
    for _ in range(warmup_runs):
        _ = model(dummy_input, verbose=False)

    # Measured runs
    latencies: list[float] = []
    peak_ram = get_process_memory_mb()

    for _ in range(num_runs):
        t_start = time.perf_counter()
        _ = model(dummy_input, verbose=False)
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
) -> BenchmarkResult:
    """Benchmark end-to-end inference on real dataset images."""
    gc.collect()
    # Read and resize images
    imgs = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(str(p))

    if not imgs:
        raise ValueError("No images found for real benchmark")

    t0 = time.perf_counter()
    _ = model(imgs[0], imgsz=imgsz, verbose=False)
    cold_start_ms = (time.perf_counter() - t0) * 1000.0

    latencies: list[float] = []
    peak_ram = get_process_memory_mb()

    for img_path in imgs:
        t_start = time.perf_counter()
        _ = model(img_path, imgsz=imgsz, verbose=False)
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
) -> BenchmarkResult:
    """Profile SAHI sliced inference on CPU."""
    gc.collect()
    dummy_img = (np.random.rand(img_size, img_size, 3) * 255).astype(np.uint8)

    t0 = time.perf_counter()
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=0.25,
        device="cpu",
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
) -> dict[str, str | int | float]:
    """Test backward pass, gradient allocation, and step latency."""
    gc.collect()
    m = model.model
    m.train()
    for p in m.parameters():
        p.requires_grad = True

    optimizer = torch.optim.SGD(m.parameters(), lr=0.001)
    latencies: list[float] = []
    peak_ram = get_process_memory_mb()

    dummy_input = torch.rand(batch_size, 3, img_size, img_size)

    def extract_loss(obj: object) -> torch.Tensor:
        total = torch.tensor(0.0, requires_grad=True)
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


def main() -> None:
    print("=" * 70)
    print("Starting Comprehensive Small Object Detection Runtime Benchmark")
    print("=" * 70)

    # 1. System Metadata
    mem = psutil.virtual_memory()
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(
        f"CPU: 12th Gen Intel Core i5-12600K "
        f"({psutil.cpu_count(logical=False)} cores, {psutil.cpu_count(logical=True)} threads)"
    )
    print(f"Total RAM: {mem.total / (1024**3):.2f} GB")
    print(f"PyTorch: {torch.__version__} | Ultralytics: {YOLO.__module__}")

    # Load Models
    print("\nLoading models: yolov8s.pt and yolo26s.pt...")
    m8 = YOLO("yolov8s.pt")
    m26 = YOLO("yolo26s.pt")

    results: dict[str, object] = {
        "hardware": {
            "cpu": "Intel Core i5-12600K (10 cores, 16 threads)",
            "gpu": "AMD Radeon RX 6700 XT 12GB (DirectML evaluated, CPU baseline active)",
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "pytorch_version": torch.__version__,
        },
        "resolution_sweep": [],
        "batch_sweep": [],
        "thread_scaling": [],
        "real_images_sweep": [],
        "sahi_sweep": [],
        "training_step": [],
    }

    # 2. Resolution Sweep (Batch 1, default threads)
    resolutions = [640, 800, 1024, 1280]
    print("\n[Sweep 1/6] Resolution Sweep (batch=1)...")
    for res in resolutions:
        print(f"  Testing Resolution {res}x{res}...")
        r8 = benchmark_inference(m8, "YOLOv8s", res, batch_size=1, num_runs=8)
        r26 = benchmark_inference(m26, "YOLO26s", res, batch_size=1, num_runs=8)
        results["resolution_sweep"].append(asdict(r8))
        results["resolution_sweep"].append(asdict(r26))
        print(
            f"    YOLOv8s:  {r8.mean_latency_ms:6.1f} ms | {r8.fps:5.2f} FPS | Peak RAM: {r8.peak_ram_mb:6.1f} MB"
        )
        print(
            f"    YOLO26s:  {r26.mean_latency_ms:6.1f} ms | {r26.fps:5.2f} FPS | Peak RAM: {r26.peak_ram_mb:6.1f} MB"
        )

    # 3. Batch Size Sweep (Resolution 800x800 - capped at 16 for sensible run duration)
    batches = [1, 2, 4, 8, 16]
    print("\n[Sweep 2/6] Batch Size Sweep (AI-TOD 800x800)...")
    for b in batches:
        print(f"  Testing Batch Size {b}...")
        r8 = benchmark_inference(m8, "YOLOv8s", 800, batch_size=b, num_runs=3)
        r26 = benchmark_inference(m26, "YOLO26s", 800, batch_size=b, num_runs=3)
        results["batch_sweep"].append(asdict(r8))
        results["batch_sweep"].append(asdict(r26))
        print(
            f"    YOLOv8s:  {r8.mean_latency_ms:7.1f} ms | {r8.fps:5.2f} FPS "
            f"({r8.mean_latency_ms/b:5.1f} ms/img) | Peak RAM: {r8.peak_ram_mb:6.1f} MB"
        )
        print(
            f"    YOLO26s:  {r26.mean_latency_ms:7.1f} ms | {r26.fps:5.2f} FPS "
            f"({r26.mean_latency_ms/b:5.1f} ms/img) | Peak RAM: {r26.peak_ram_mb:6.1f} MB"
        )

    # 4. CPU Thread Scaling Sweep (800x800, Batch 1)
    thread_counts = [1, 2, 4, 8, 16]
    print("\n[Sweep 3/6] Thread Scaling Sweep (800x800, batch=1)...")
    for th in thread_counts:
        print(f"  Testing Thread Count {th}...")
        r8 = benchmark_inference(m8, "YOLOv8s", 800, batch_size=1, num_runs=5, threads=th)
        r26 = benchmark_inference(m26, "YOLO26s", 800, batch_size=1, num_runs=5, threads=th)
        results["thread_scaling"].append(asdict(r8))
        results["thread_scaling"].append(asdict(r26))
        print(f"    YOLOv8s (th={th:2d}): {r8.mean_latency_ms:6.1f} ms | {r8.fps:5.2f} FPS")
        print(f"    YOLO26s (th={th:2d}): {r26.mean_latency_ms:6.1f} ms | {r26.fps:5.2f} FPS")

    torch.set_num_threads(16)

    # 5. Real Drone Imagery Sweep
    real_sample_dir = Path("data/visdrone/sample_20")
    if real_sample_dir.exists():
        real_images = list(real_sample_dir.glob("*.jpg"))
        if real_images:
            print(f"\n[Sweep 4/6] Real Drone Imagery Sweep ({len(real_images)} images, imgsz=800)...")
            rr8 = benchmark_real_images(m8, "YOLOv8s", real_images, imgsz=800)
            rr26 = benchmark_real_images(m26, "YOLO26s", real_images, imgsz=800)
            results["real_images_sweep"].append(asdict(rr8))
            results["real_images_sweep"].append(asdict(rr26))
            print(
                f"    YOLOv8s:  {rr8.mean_latency_ms:6.1f} ms | {rr8.fps:5.2f} FPS | Peak RAM: {rr8.peak_ram_mb:6.1f} MB"
            )
            print(
                f"    YOLO26s:  {rr26.mean_latency_ms:6.1f} ms | {rr26.fps:5.2f} FPS | Peak RAM: {rr26.peak_ram_mb:6.1f} MB"
            )

    # 6. SAHI Tiled Inference Sweep
    print("\n[Sweep 5/6] SAHI Tiled Inference Sweep (512x512 slice, 0.2 overlap)...")
    for res in [800, 1280]:
        print(f"  Testing SAHI on {res}x{res}...")
        s8 = benchmark_sahi("yolov8s.pt", "YOLOv8s", res, slice_size=512, overlap_ratio=0.2, num_runs=2)
        s26 = benchmark_sahi("yolo26s.pt", "YOLO26s", res, slice_size=512, overlap_ratio=0.2, num_runs=2)
        results["sahi_sweep"].append(asdict(s8))
        results["sahi_sweep"].append(asdict(s26))
        print(
            f"    YOLOv8s SAHI ({res}x{res}): {s8.mean_latency_ms:7.1f} ms | {s8.fps:5.2f} FPS | Peak RAM: {s8.peak_ram_mb:6.1f} MB"
        )
        print(
            f"    YOLO26s SAHI ({res}x{res}): {s26.mean_latency_ms:7.1f} ms | {s26.fps:5.2f} FPS | Peak RAM: {s26.peak_ram_mb:6.1f} MB"
        )

    # 7. Dummy Training / Backward Pass Step
    print("\n[Sweep 6/6] Dummy Training Step & Memory Check (800x800)...")
    t8_b2 = benchmark_dummy_training(m8, "YOLOv8s", img_size=800, batch_size=2)
    t26_b2 = benchmark_dummy_training(m26, "YOLO26s", img_size=800, batch_size=2)
    results["training_step"].extend([t8_b2, t26_b2])
    print(f"  YOLOv8s  (batch=2): {t8_b2['mean_step_ms']} ms/step | Peak RAM: {t8_b2['peak_ram_mb']} MB")
    print(f"  YOLO26s  (batch=2): {t26_b2['mean_step_ms']} ms/step | Peak RAM: {t26_b2['peak_ram_mb']} MB")

    # Output JSON summary
    out_dir = Path("benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "runtime_benchmark_raw.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw benchmark data saved to {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
