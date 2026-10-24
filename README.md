# Small Object Detection (AI-TOD-v2 Benchmark Suite)

Benchmarking and ablation study suite for tiny and small object detection on aerial remote sensing imagery (AI-TOD-v2), contrasting YOLOv8s and YOLO26s baselines with architectural ablations (P2 feature stride, SAHI sliced inference, and Normalized Gaussian Wasserstein Distance / NWD loss).

## Key Characteristics of AI-TOD-v2

Empirical analysis across 376,625 annotations demonstrates severe micro-scale target concentrations:
- **Median scale ($\sqrt{w \times h}$):** 11.83 px (mean: 12.67 px).
- **86.54%** of instances measure $\le 16$ px ($AP_{vt} \le 8$ px, $AP_t \in (8, 16]$ px).
- **98.20%** of instances measure $< 32$ px (classified simply as "Small" in MS COCO).
- See [`docs/dataset_analysis.md`](docs/dataset_analysis.md) for full scale diagnostic derivations, feature map stride collapse analysis, and empirical visualizations.

---

## Installation & Setup

Manage dependencies with `uv`:

```powershell
# Sync project virtual environment
uv sync
```

---

## Unified CLI (`sod`)

The command-line interface provides end-to-end tooling for data ingestion, benchmarking, training, and evaluation:

```powershell
# Top-level commands
uv run sod --help
```

### 1. Data Ingestion & Preparation

```powershell
# Download datasets (VisDrone2019-DET and AI-TOD-v2)
uv run sod data download --dataset aitod-v2

# Run exploratory data analysis and generate diagnostic plots in assets/eda/
uv run sod data eda

# Generate stratified development split (1,000 train, 200 val) and YOLO labels
uv run sod data prepare --train-size 1000 --val-size 200
```

### 2. Runtime & Hardware Profiling

```powershell
# Benchmark YOLOv8s vs. YOLO26s latency, throughput, SAHI tiling, and RAM usage
uv run sod bench runtime --img-size 800
```
Detailed performance profiling is documented in [`docs/benchmarks/runtime_profiling.md`](docs/benchmarks/runtime_profiling.md).

---

## Project Structure

```text
small-object-detection/
├── assets/
│   └── eda/                  # Publication-grade diagnostic figures
├── configs/
│   ├── default.yaml          # Default dataset paths and evaluation scale bins
│   └── aitod_mini.yaml       # 1,000-image stratified development dataset config
├── docs/
│   ├── dataset_analysis.md   # Scale diagnostics, stride collapse, and IoU sensitivity
│   └── benchmarks/
│       └── runtime_profiling.md # Latency, batch scaling, and VRAM/RAM analysis
├── src/
│   ├── bench/                # Hardware and latency profiling logic
│   ├── data/                 # Ingestion, EDA, sampling, and COCO-to-YOLO conversion
│   ├── eval/                 # Dual-surface evaluation (COCO mAP & AI-TOD micro bins)
│   ├── infer/                # Inference engine and SAHI slicing wrappers
│   ├── models/               # Model configurations and custom head definitions
│   ├── utils/                # UI presentation helpers and unified plotting styles
│   └── cli.py                # Unified Typer CLI entrypoint (`sod`)
├── scripts/                  # Lightweight entry point shims
├── pyproject.toml            # Project dependencies and script registration
└── README.md
```
