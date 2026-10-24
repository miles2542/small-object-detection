## Installation & Setup

Manage dependencies with `uv`:

```powershell
# Sync project virtual environment
uv sync

# Recommended: Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

Once activated, run `sod` directly:
```powershell
sod --help
```

*Tip for Windows users:* You can also run the root wrapper directly without activating the environment:
```powershell
.\sod --help
```

For platform-specific execution profiles (Local CPU, RTX 3060 CUDA, Kaggle Dual T4) and downstream track integration, refer to the [Team Quickstart & Handoff Guide](docs/team_quickstart.md).

---

## Unified CLI (`sod`)

The command-line interface provides end-to-end tooling for data ingestion, benchmarking, training, and evaluation. Detailed walkthrough and parameter flags are documented in [`docs/team_quickstart.md`](docs/team_quickstart.md).

```powershell
# Top-level commands
sod --help
```

### 1. Data Ingestion & Preparation

```powershell
# Primary: Download full AI-TOD-v2 dataset splits for complete training sweeps
sod data download --dataset aitod-v2 --splits train val

# Prepare full dataset (writes configs/aitod_full.yaml)
sod data prepare --train-size 0 --val-size 0 --output-dir data/ai-tod-v2/splits/full --config-yaml configs/aitod_full.yaml

# Quick local testing: Download sample subset and generate 1,000-image development split
sod data download --dataset aitod-v2 --sample-only --sample-count 500
sod data prepare --train-size 1000 --val-size 200 --output-dir data/ai-tod-v2/splits/mini --config-yaml configs/aitod_mini.yaml

# Run exploratory data analysis and generate diagnostic plots in assets/eda/
sod data eda
```

### 2. Runtime & Hardware Profiling

```powershell
# Benchmark YOLOv8s vs. YOLO26s latency, throughput, SAHI tiling, and RAM usage
sod bench runtime --img-size 800
```
Detailed performance profiling is documented in [`docs/benchmarks/runtime_profiling.md`](docs/benchmarks/runtime_profiling.md).

### 3. Model Training & Cloud GPU Execution

The training CLI auto-detects compute hardware: selecting CUDA GPU (`0`) and batch size 16 if available, otherwise falling back to CPU and batch size 4.

```powershell
# Primary: Cloud Kaggle Dual-T4 GPU training ($136.4x speedup, 1,000 images in ~16s/epoch)
sod bench kaggle scaffold --epochs 10 --batch-size 16
sod bench kaggle push
sod bench kaggle status
sod bench kaggle output

# Local CUDA training (e.g. RTX 3060 laptop)
sod train run --model yolo26s --config configs/aitod_full.yaml --epochs 10

# Local CPU sanity check
sod train run --model yolov8s --config configs/aitod_mini.yaml --epochs 1 --fraction 0.05
```
