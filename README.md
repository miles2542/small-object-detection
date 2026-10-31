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

### 4. Size-Based Evaluation

```powershell
sod eval run --help

# Genuine eight-class AI-TOD checkpoint: export predictions and evaluate
sod eval run --weights runs/train/yolov8s_aitod_mini/weights/best.pt --data configs/aitod_mini.yaml --split val --output-dir runs/eval/yolov8s_trained

# Reuse saved predictions with their exact COCO ground-truth JSON
sod eval run --data runs/eval/input_audit/mini_val_ground_truth.json --predictions runs/eval/phase2_yolo26s_smoke/predictions.json --smoke --output-dir runs/eval/yolo26s_json
```

Use a fresh output directory for each run. `--smoke` explicitly permits pretrained partial-vocabulary diagnostics; these do not establish trained AI-TOD baseline performance. `--confidence` changes fixed-confidence precision/recall while leaving the AP ranking unchanged. Saved-prediction mode needs no model inference or local image directory.

The CLI saves full image coverage, provenance, AP/AR, per-class size metrics and PR tensors. See the [evaluation reports](docs/evaluation/README.md) for the protocol, validation evidence, figures and benchmark handoff status.

### 5. Saved-Run Analysis and Handoff

```powershell
python -m src.eval.analysis --help
# Compare two complete CLI runs; no inference is repeated.
python -m src.eval.analysis --run YOLOv8s=runs/eval/v8_full --run YOLO26s=runs/eval/26_full --output-dir runs/eval/full_analysis --image-root data/ai-tod-v2/splits/full/images/val
```

Run `sod eval run` first to create the two input directories. Analysis checks matching GT/protocol/export settings and reproduces saved metrics, then writes AP comparisons, signed class-controlled size gaps, support, optional error crops, `analysis.json` and `comparison.md`. Smoke inputs stay explicitly labeled; other inputs require training-provenance review. See the [benchmark status and handoff](docs/evaluation_baselines.md) and [Phase 5 report](docs/evaluation/05_analysis_and_handoff.md).
