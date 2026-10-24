# Team quickstart and handoff guide

Operational guide for environment setup, data preparation, baseline training execution profiles, and downstream evaluation tracks.

---

## 1. Environment setup

Install dependencies using `uv`:

```powershell
# Synchronize project virtual environment
uv sync
```

### CLI wrappers
On Windows (PowerShell or CMD), invoke the root CLI wrapper directly:
```powershell
.\sod --help
```

On Linux or macOS:
```bash
uv run sod --help
# or directly via module entrypoint:
uv run python -m src.cli --help
```

Pretrained weights (`yolov8s.pt`, `yolo26s.pt`) resolve automatically from `weights/pretrained/` or download on first invocation.

---

## 2. Dataset preparation

AI-TOD-v2 contains 376,625 annotations across 14,018 aerial images ($800 \times 800$ resolution), with 86.54% of targets measuring $\le 16$ px.

### Step 1: Download annotations and imagery
```powershell
# Fast sample download for immediate testing
.\sod data download --dataset aitod-v2 --sample-only

# Or download full splits for complete training sweeps
.\sod data download --dataset aitod-v2 --splits train val
```

### Step 2: Generate stratified mini split
Generate the balanced development split (1,000 train images, 200 validation images) preserving category ratios and micro-scale bins ($AP_{vt}, AP_t, AP_s, AP_m$):
```powershell
.\sod data prepare --train-size 1000 --val-size 200
```
This command:
- Filters 40+ zero-dimension degenerate boxes ($w \le 0$ or $h \le 0$) from raw COCO annotations.
- Exports sanitized YOLO format label files (`.txt`) to `data/ai-tod-v2/splits/mini/`.
- Writes the dataset definition to `configs/aitod_mini.yaml`.

---

## 3. Execution profiles

Select the execution profile matching available compute:

### Profile A: Local CPU or Apple silicon (debugging and sanity checks)
```powershell
.\sod train run --model yolov8s --config configs/aitod_mini.yaml --epochs 1 --batch-size 2 --device cpu
```
- Throughput: $\sim 0.45\text{ imgs/s}$.
- Memory: $\sim 1.5\text{ GB}$ system RAM.
- Use case: Code path verification, config sanity, dataloader checks.

### Profile B: Local CUDA GPU (e.g., RTX 3060 laptop, 6GB-12GB VRAM)
```powershell
.\sod train run --model yolo26s --config configs/aitod_mini.yaml --epochs 10 --batch-size 8 --device 0
```
- Throughput: $\sim 45\text{--}65\text{ imgs/s}$.
- Memory: $\sim 4.2\text{ GB}$ VRAM at batch size 8.
- Use case: Multi-epoch baseline training, rapid architecture iterations.

### Profile C: Headless cloud GPU (Kaggle dual Tesla T4, 2x 15GB VRAM)
Requires Kaggle API token configured at `~/.kaggle/kaggle.json`.

```powershell
# 1. Generate Kaggle kernel runner script and metadata
.\sod bench kaggle scaffold --epochs 10 --batch-size 8

# 2. Push kernel to start remote Dual T4 training
.\sod bench kaggle push

# 3. Poll kernel execution status
.\sod bench kaggle status

# 4. Download trained weights and run artifacts
.\sod bench kaggle output --output-dir runs/kaggle
```
- Throughput: $61.36\text{ imgs/s}$ (YOLOv8s at batch size 16).
- Memory: $5,110\text{ MB}$ VRAM ($34.1\%$ utilization).
- Use case: 100-epoch full runs, multi-model factorial sweeps.

---

## 4. Checkpoints and downstream integration

### Checkpoint storage locations
Local training runs output checkpoints and metrics to `runs/train/`:
- Best checkpoint: `runs/train/<model>_aitod_mini/weights/best.pt`
- Last checkpoint: `runs/train/<model>_aitod_mini/weights/last.pt`
- Training metrics and logs: `runs/train/<model>_aitod_mini/results.csv`
- Verification curves: `runs/train/<model>_aitod_mini/confusion_matrix.png`

Kaggle outputs download to `runs/kaggle/` via `.\sod bench kaggle output`.

### Loading checkpoints in Python
```python
from ultralytics import YOLO

# Load best checkpoint from training run
model = YOLO("runs/train/yolov8s_aitod_mini/weights/best.pt")

# Run inference on native 800x800 test image
results = model.predict(
    source="data/ai-tod-v2/splits/mini/images/val",
    imgsz=800,
    conf=0.25,
    device="cpu",  # or 0 for CUDA
)
```

---

## 5. Downstream milestone integration

Baseline checkpoints and splits feed into the following subsequent tracks:

### Size-based evaluation track
Standard aggregate mAP masks microscopic object failure. Downstream evaluation pipelines load baseline weights and execute dual-surface evaluation:
- Run inference across the validation split.
- Pipe prediction bounding boxes into the AI-TOD evaluator.
- Decompose performance across micro-scale intervals:
  - Very Tiny ($AP_{vt}$): Area $\in (2, 8]$ px.
  - Tiny ($AP_t$): Area $\in (8, 16]$ px.
  - Small ($AP_s$): Area $\in (16, 32]$ px.
  - Medium ($AP_m$): Area $\in (32, 64]$ px.

### P2 high-resolution stride track
AI-TOD median target size (11.83 px) collapses to $1.48 \times 1.48$ cells at P3 (stride 8) and $< 1$ cell at P4/P5.
- Downstream architecture modifications introduce a stride-4 ($P_2$) detection head to preserve $2.96 \times 2.96$ feature representation cells for median targets.
- Train the modified P2 network against `configs/aitod_mini.yaml` using baseline checkpoints as initialization.

### SAHI sliced inference track
Slicing divides high-resolution $800 \times 800$ frames into smaller patches (e.g. $320 \times 320$, overlap 0.25) to amplify relative object pixel dimensions before NMS/NMM merging.
- Use baseline `best.pt` models inside SAHI inference wrappers:
```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="runs/train/yolov8s_aitod_mini/weights/best.pt",
    confidence_threshold=0.25,
    device="cpu",
)

result = get_sliced_prediction(
    "data/ai-tod-v2/splits/mini/images/val/sample.jpg",
    detection_model,
    slice_height=320,
    slice_width=320,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25,
)
```

### Loss ablation track (NWD)
Tiny bounding box IoU collapses rapidly under 1-to-2 pixel spatial deviations, starving standard anchor assigners of positive labels.
- Downstream loss ablations introduce Normalized Gaussian Wasserstein Distance (NWD) loss to provide smooth distance gradients for sub-16px boxes.
- Benchmark NWD loss variants against baseline CIoU / DFL loss runs.

### Reporting and synthesis track
- Extract scalar metrics from `runs/train/<run_id>/results.csv`.
- Aggregate precision-recall curves, confusion matrices, and validation batches from run directories for publication figures.
