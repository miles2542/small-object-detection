# Team quickstart and handoff guide

Operational guide for environment setup, data preparation, baseline training execution profiles, and downstream evaluation tracks.

---

## 1. Environment setup

Install dependencies using `uv`:

```powershell
# Synchronize project virtual environment
uv sync
```

### Virtual environment activation (Recommended)

To ensure tools, Python interpreters, and CLI scripts operate cleanly without path issues across all terminals:

```powershell
# Windows PowerShell (Recommended)
.venv\Scripts\Activate.ps1

# Windows Command Prompt (CMD)
.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```

Once activated, invoke commands directly:
```powershell
sod --help
```

*Note for Windows users:* If you do not activate the environment, invoke the native root wrapper directly:
```powershell
.\sod --help
```

Pretrained weights (`yolov8s.pt`, `yolo26s.pt`) resolve automatically from `weights/pretrained/` or download autonomously from Ultralytics releases on first invocation.

---

## 2. Kaggle cloud GPU API setup

For heavy multi-epoch training sweeps and ablation matrices on Kaggle's Dual NVIDIA Tesla T4 GPUs (30 hours/week free tier):

1. Log into [Kaggle](https://www.kaggle.com/) $\rightarrow$ **Settings** $\rightarrow$ scroll to **API *(The new, recommended one, not the legacy)*** $\rightarrow$ click **Create New Token**.
2. Save your API token:
   - On Windows PowerShell:
     ```powershell
     $kaggleDir = "$env:USERPROFILE\.kaggle"
     if (-not (Test-Path $kaggleDir)) { New-Item -ItemType Directory -Path $kaggleDir -Force | Out-Null }
     Set-Content -Path "$kaggleDir\access_token" -Value "YOUR_KGAT_TOKEN_HERE" -NoNewline
     ```
   - On Linux / macOS:
     ```bash
     mkdir -p ~/.kaggle && echo "<YOUR_KGAT_TOKEN_HERE>" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
     ```
3. Test connectivity:
   ```powershell
   kaggle competitions list
   ```

---

## 3. Dataset preparation

AI-TOD-v2 contains 376,625 annotations across 14,018 aerial images ($800 \times 800$ resolution), with 86.54% of targets measuring $\le 16$ px.

### Step 1: Download imagery and annotations

The pipeline downloads directly from verified Google Drive mirrors with zero manual token configuration:

```powershell
# Primary: Download full AI-TOD-v2 dataset splits for complete training sweeps
sod data download --dataset aitod-v2 --splits train val

# Quick verification: Download small 500-image sample subset
sod data download --dataset aitod-v2 --sample-only --sample-count 500
```

### Step 2: Generate dataset splits and YOLO labels

```powershell
# Primary: Prepare full dataset for 100% benchmark runs (writes configs/aitod_full.yaml)
sod data prepare --train-size 0 --val-size 0 --output-dir data/ai-tod-v2/splits/full --config-yaml configs/aitod_full.yaml

# Local development: Generate stratified 1,000-train / 200-val development split
sod data prepare --train-size 1000 --val-size 200 --output-dir data/ai-tod-v2/splits/mini --config-yaml configs/aitod_mini.yaml
```

The converter automatically:
- Filters degenerate zero-dimension annotations ($w \le 0$ or $h \le 0$) to protect loss functions from NaN divergence.
- Normalizes coordinates $[x, y, w, h]$ to bounded $[0.0, 1.0]$.
- Harmonizes category indices to standard 0-indexed AI-TOD IDs (`0: airplane, 1: bridge, 2: storage-tank, 3: ship, 4: swimming-pool, 5: vehicle, 6: person, 7: wind-mill`).

---

## 4. Execution profiles

The training engine auto-detects hardware: `--device` automatically selects CUDA GPU (`0`) if available, falling back to `cpu`. Default batch size auto-scales to `16` on GPU and `4` on CPU.

### Profile A: Cloud Kaggle Dual-T4 GPU (Primary / Recommended)
Trains 1,000 images in **16.4 seconds per epoch** ($136.4\times$ speedup over high mid-range desktop CPU).

```powershell
# 1. Generate Kaggle kernel runner script and metadata (batch 16, dual GPU enabled)
sod bench kaggle scaffold --epochs 10 --batch-size 16

# 2. Push kernel to start remote Dual T4 training
sod bench kaggle push

# 3. Poll kernel execution status
sod bench kaggle status

# 4. Download trained weights and run artifacts
sod bench kaggle output --output-dir runs/kaggle
```

### Profile B: Local CUDA GPU (e.g., RTX 3060 Laptop)
Auto-detects GPU and batch size 16:
```powershell
sod train run --model yolo26s --config configs/aitod_mini.yaml --epochs 10
```

### Profile C: Local CPU or Apple Silicon (Fast sanity checks)
```powershell
sod train run --model yolov8s --config configs/aitod_mini.yaml --epochs 1 --fraction 0.05
```

---

## 5. Next tasks & member assignments

Milestone 1 delivers the verified foundation, EDA, and baseline pipeline. Subsequent project tasks are distributed across team members as follows:

### Member 1
- **Completed (M1):** Ingestion pipeline, empirical EDA (`docs/dataset_analysis.md`), stratified mini-split converter (`src/data/`), baseline training engine (`src/models/train.py`), Kaggle remote runner orchestration, and repository curation.
- **Ongoing Responsibilities:** Code review and milestone integration across all teammate PRs, remote Kaggle cloud compute coordination, and overall system architecture integrity.

### Member 2 (Size-Based Dual-Surface Evaluation)
- **Role:** Build and integrate the fine-grained micro-scale evaluation module in `src/eval/`.
- **Key Deliverable:** Standard aggregate $mAP_{50:95}$ obscures microscopic target loss. Implement evaluation pipelines that ingest baseline predictions and calculate AP across official AI-TOD scale bins:
  - Very Tiny ($AP_{vt}$): $\text{scale} \le 8$ px.
  - Tiny ($AP_t$): $8 < \text{scale} \le 16$ px.
  - Small ($AP_s$): $16 < \text{scale} \le 32$ px.
  - Medium ($AP_m$): $32 < \text{scale} \le 64$ px.
- **Reference Checkpoints:** `runs/train/yolov8s_aitod_mini/weights/best.pt` and `yolo26s_aitod_mini/weights/best.pt`.

### Member 3 (Architecture Modification — P2 High-Resolution Stride Head)
- **Role:** Modify model architectures in `src/models/` to introduce a $P_2$ ($4\times$ downsampling stride) feature pyramid head.
- **Key Deliverable:** Our EDA proved that median targets (11.83 px) collapse to $1.48 \times 1.48$ cells at $P_3$ and $<1$ cell at $P_4/P_5$. Implement a high-resolution $P_2$ detection head to preserve $2.96 \times 2.96$ spatial cells. Train and compare against baseline YOLOv8s/YOLO26s.

### Member 4 (Slicing-Aided Hyper Inference — SAHI)
- **Role:** Benchmark and optimize sliced patch inference in `src/infer/`.
- **Key Deliverable:** Tune SAHI slicing windows (e.g. $512 \times 512$ with 0.2 overlap) on $800 \times 800$ frames. Evaluate the trade-off between localization recovery on micro targets versus the established $\sim 3.5\times$ computational latency overhead.

### Member 5 (Loss Function Ablation — NWD)
- **Role:** Implement Normalized Gaussian Wasserstein Distance (NWD) loss in `src/models/`.
- **Key Deliverable:** Small bounding box IoU drops from 1.0 to 0.47 under a 2-pixel shift. Integrate NWD loss to provide smooth distance gradients for sub-16px boxes, replacing or supplementing CIoU/DFL, and benchmark convergence improvements.

### Member 6 (Benchmarking Synthesis, Final Report & Defense)
- **Role:** Consolidate cross-model ablation metrics, generate comparative tables, and prepare project defense materials.
- **Key Deliverable:** Aggregate validation curves, confusion matrices, and results from `runs/train/` and `runs/kaggle/`. Author the final academic paper and presentation slides comparing standard YOLOv8s vs. YOLO26s with $P_2$, SAHI, and NWD ablations.

---

## 6. Git branch and pull request workflow

The `main` branch is protected against direct pushes to safeguard shared baseline code and configurations. All contributions must be completed on a dedicated feature branch and merged via GitHub Pull Requests.

> [!NOTE]
> Teammates are not expected to worry about manual code formatting or running linters before pushing. Work freely on your task; the lead orchestrator reviews and handles lint passes prior to merging.

### Step 1: Create a feature branch
Before making edits, create a new branch from `main`:

- **Using VS Code GUI:**
  1. Click the current branch name (`main`) in the bottom-left status bar (or open the **Source Control** tab `Ctrl+Shift+G` $\rightarrow$ click `...` $\rightarrow$ **Branch** $\rightarrow$ **Create Branch...**).
  2. Enter a descriptive branch name, e.g.:
     - Member 2: `feat/m2-evaluation-coco`
     - Member 3: `feat/m3-p2-stride-head`
     - Member 4: `feat/m4-sahi-inference`
     - Member 5: `feat/m5-nwd-loss`
     - Member 6: `feat/m6-benchmarking-report`
- **Using VS Code Terminal:**
  ```powershell
  git checkout -b feat/m2-evaluation-coco
  ```

### Step 2: Implement and commit your changes
Work normally in your branch:

- **Using VS Code GUI:**
  1. Open the **Source Control** view (`Ctrl+Shift+G`).
  2. Click the `+` icon next to modified files to stage them.
  3. Enter a clear commit message in the text box above (e.g. `feat: add COCO evaluation script for micro-scale metrics`).
  4. Click **Commit** (or press `Ctrl+Enter`).
- **Using VS Code Terminal:**
  ```powershell
  git add .
  git commit -m "feat: add COCO evaluation script for micro-scale metrics"
  ```

### Step 3: Push your branch to GitHub
Once your work is committed and ready for review:

- **Using VS Code GUI:**
  - Click the blue **Publish Branch** button in the Source Control panel or status bar.
- **Using VS Code Terminal:**
  ```powershell
  git push -u origin feat/m2-evaluation-coco
  ```

### Step 4: Open a Pull Request on GitHub Web
1. Open the repository in your browser: [https://github.com/miles2542/small-object-detection](https://github.com/miles2542/small-object-detection)
2. You will see a banner at the top: **"feat/... had recent pushes ... Compare & pull request"**. Click the green button.
   *(If not shown, go to **Pull requests** tab $\rightarrow$ click **New pull request** $\rightarrow$ select `base: main` and `compare: your-branch`).*
3. Provide a brief summary of what was implemented or tested.
4. Click **Create pull request**. The lead will review and merge into `main`.
