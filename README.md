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

For zero-setup CLI execution on Windows, invoke `.\sod` directly from the repository root:
```powershell
.\sod --help
```

For platform-specific execution profiles (CPU, RTX 3060 CUDA, Kaggle Dual T4) and downstream track integration, refer to the [Team Quickstart & Handoff Guide](docs/team_quickstart.md).

---

## Unified CLI (`sod`)

The command-line interface provides end-to-end tooling for data ingestion, benchmarking, training, and evaluation. Detailed walkthrough and parameter flags are documented in [`docs/team_quickstart.md`](docs/team_quickstart.md).

```powershell
# Top-level commands
.\sod --help
```

### 1. Data Ingestion & Preparation

```powershell
# Download datasets (VisDrone2019-DET and AI-TOD-v2 via direct Google Drive mirror)
.\sod data download --dataset aitod-v2 --sample-only

# Run exploratory data analysis and generate diagnostic plots in assets/eda/
.\sod data eda

# Generate stratified development split (1,000 train, 200 val) and YOLO labels
.\sod data prepare --train-size 1000 --val-size 200
```

### 2. Runtime & Hardware Profiling

```powershell
# Benchmark YOLOv8s vs. YOLO26s latency, throughput, SAHI tiling, and RAM usage
.\sod bench runtime --img-size 800
```
Detailed performance profiling is documented in [`docs/benchmarks/runtime_profiling.md`](docs/benchmarks/runtime_profiling.md) and [`docs/benchmarks/training_throughput_profiling.md`](docs/benchmarks/training_throughput_profiling.md).

### 3. Model Training & Cloud GPU Execution

```powershell
# Local baseline training (CPU / CUDA)
.\sod train run --model yolov8s --config configs/aitod_mini.yaml --epochs 10 --device cpu

# Headless Kaggle Dual-T4 GPU execution
.\sod bench kaggle scaffold --epochs 10 --batch-size 8
.\sod bench kaggle push
.\sod bench kaggle status
.\sod bench kaggle output
```

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
│   ├── team_quickstart.md    # Team onboarding, compute profiles, and downstream handoff
│   ├── dataset_analysis.md   # Scale diagnostics, stride collapse, and IoU sensitivity
│   └── benchmarks/
│       ├── runtime_profiling.md             # Latency, batch scaling, and VRAM/RAM analysis
│       └── training_throughput_profiling.md # CPU vs. Kaggle Dual T4 empirical profiling
├── kaggle/
│   └── kernels/aitod_train/  # Headless Dual T4 Kaggle runner scripts and metadata
├── src/
│   ├── bench/                # Hardware latency and Kaggle remote runner logic
│   ├── data/                 # Ingestion, EDA, sampling, and COCO-to-YOLO conversion
│   ├── eval/                 # Dual-surface evaluation (COCO mAP & AI-TOD micro bins)
│   ├── infer/                # Inference engine and SAHI slicing wrappers
│   ├── models/               # Model configurations, training pipeline, and custom heads
│   ├── utils/                # UI presentation helpers and unified plotting styles
│   └── cli.py                # Unified Typer CLI entrypoint (`sod`)
├── weights/                  # Local pretrained checkpoints (.pt)
├── sod.ps1                   # Root PowerShell CLI wrapper
├── sod.cmd                   # Root CMD CLI wrapper
├── pyproject.toml            # Project dependencies and script registration
└── README.md
```
