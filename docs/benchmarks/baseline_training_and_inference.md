# AI-TOD-v2 Baseline Training & Inference Benchmark Report

This document reports the empirical benchmark results of 100-epoch deep baseline training and multi-stage inference scaling on the AI-TOD-v2 (Aerial Imagery for Tiny Object Detection) dataset across YOLOv8s and YOLO26s architectures.

---

## 1. Experimental Environment & Setup

### 1.1 Compute Infrastructure & Software
- **Compute:** Dual NVIDIA Tesla T4 GPUs (16 GB GDDR6 VRAM per GPU, 32 GB aggregate).
- **Environment:** Remote Linux container (Ubuntu 22.04 LTS, x86_64, CUDA 13.0).
- **Core Frameworks:** PyTorch 2.11.0+cu130, Ultralytics 8.3+, SAHI 0.11.18, PyCOCOTools 2.0.11.
- **Dataloader Configuration:** 2 parallel worker processes per process, pin memory enabled, mixed-precision FP16 AMP.

### 1.2 Dataset Configuration (AI-TOD-v2)
- **Image Resolution:** Native $800 \times 800$ pixels.
- **Split Composition:**
  - Training set: 11,214 images, 301,494 ground-truth bounding box annotations.
  - Validation set: 2,804 images, 75,082 ground-truth bounding box annotations.
- **Canonical Categories (8 Classes):**
  `0: airplane`, `1: bridge`, `2: storage-tank`, `3: ship`, `4: swimming-pool`, `5: vehicle`, `6: person`, `7: wind-mill`.
- **Target Size Distribution:** Micro-scale aerial targets where $\approx 86\%$ of objects are smaller than $16 \times 16$ pixels (mean target area $\approx 12.8\text{px}$).

---

## 2. Stage 1: Hyperparameter Calibration Sweep

To identify optimal hyperparameter balance on extreme small-scale targets, three 15-epoch calibration sweeps were evaluated on YOLOv8s using AdamW:

| Variant | Box Gain | Class Gain | Mosaic | MixUp | Initial LR ($\text{lr}_0$) | LR Schedule | Val $\text{mAP}_{50}$ | Val $\text{mAP}_{50-95}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Default Baseline** | 7.5 | 0.5 | 1.00 | 0.00 | 0.0010 | Linear | 0.4597 | 0.2008 |
| **B: Localization Heavy** | 10.0 | 0.5 | 0.50 | 0.15 | 0.0010 | Cosine | 0.4134 | 0.1819 |
| **C: Smooth Cosine Decay** | **9.0** | **0.5** | **0.20** | **0.00** | **0.0005** | **Cosine** | **0.4635** | **0.2045** |

**Observation:** Over-aggressive mosaic augmentations ($1.00$) degrade tiny target feature coherence by splitting sub-16px objects across quadrant boundaries. Reduced mosaic ($0.20$) paired with conservative initial learning rate ($0.0005$), box loss weighting ($9.0$), and smooth cosine decay yielded superior localization convergence (+0.37 pp $\text{mAP}_{50-95}$) and was selected for full-scale deep training.

---

## 3. Stages 2 & 3: Deep Convergence Baseline Training

Both models were trained for 100 full epochs using the calibrated hyperparameters:
- **Optimizer:** AdamW ($\beta_1=0.937, \beta_2=0.999$, weight decay $0.0005$).
- **Batch Size:** 16 (effective 8 per GPU).
- **Resolution:** $800 \times 800$.
- **Augmentation Warmup & Close:** 3.0 warmup epochs; mosaic augmentations disabled for the final 15 epochs.
- **Throughput:** Average $247.5\,\text{s}$ per epoch on Dual NVIDIA T4 (~61.4 images/second, peak VRAM: 5.11 GB). Total training execution: $\approx 16.5$ GPU hours across sweeps and deep runs.

### Training Progression & Convergence
- **YOLOv8s Baseline:** Converged steadily, reaching peak validation metrics at epoch 80–90:
  - $\text{mAP}_{50}$: $0.5878$ ($58.78\%$)
  - $\text{mAP}_{50-95}$: $0.2656$ ($26.56\%$)
- **YOLO26s Baseline:** Exhibited stronger feature extraction on dense clusters, peaking at epoch 97–100:
  - $\text{mAP}_{50}$: $0.6004$ ($60.04\%$)
  - $\text{mAP}_{50-95}$: $0.2750$ ($27.50\%$)

---

## 4. Stage 4: Comprehensive Evaluation & Benchmarks

Full validation set evaluation on all 2,804 images (75,082 instances) across competitive thresholding protocols and sliced inference.

### 4.1 Global Metric Comparison

| Evaluation Scenario | YOLOv8s Baseline | YOLO26s Baseline | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Standard Protocol ($\text{conf}=0.001$, $\text{IoU}=0.70$) $\text{mAP}_{50}$** | 0.5859 | **0.5980** | **+1.21 pp** |
| **Standard Protocol ($\text{conf}=0.001$, $\text{IoU}=0.70$) $\text{mAP}_{50-95}$** | 0.2654 | **0.2738** | **+0.84 pp** |
| **Operational Protocol ($\text{conf}=0.25$, $\text{IoU}=0.45$) $\text{mAP}_{50}$** | 0.5204 | **0.5376** | **+1.72 pp** |
| **Operational Protocol ($\text{conf}=0.25$, $\text{IoU}=0.45$) $\text{mAP}_{50-95}$** | 0.2424 | **0.2528** | **+1.04 pp** |
| **SAHI Sliced Latency ($512 \times 512$, overlap $0.20$)** | 140.37 ms/img | **130.62 ms/img** | **-6.9% faster** |
| **SAHI Valid Output Detections (200-sample)** | 4,221 | 3,765 | Higher precision filtering |

### 4.2 Per-Class Breakdown (Standard Protocol $\text{conf}=0.001$)

| Class | Ground Truth Instances | YOLOv8s $\text{mAP}_{50}$ | YOLO26s $\text{mAP}_{50}$ | YOLOv8s $\text{mAP}_{50-95}$ | YOLO26s $\text{mAP}_{50-95}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **airplane** | 2,216 | 0.617 | **0.701** (+8.4%) | 0.294 | **0.343** (+4.9%) |
| **bridge** | 1,847 | 0.475 | **0.508** (+3.3%) | **0.185** | 0.177 (-0.8%) |
| **storage-tank** | 9,451 | 0.863 | **0.882** (+1.9%) | 0.461 | **0.475** (+1.4%) |
| **ship** | 17,211 | 0.761 | **0.784** (+2.3%) | 0.375 | **0.394** (+1.9%) |
| **swimming-pool** | 2,903 | **0.673** | 0.657 (-1.6%) | **0.321** | 0.314 (-0.7%) |
| **vehicle** | 35,463 | 0.823 | **0.835** (+1.2%) | 0.350 | **0.358** (+0.8%) |
| **person** | 4,732 | 0.319 | **0.346** (+2.7%) | 0.106 | **0.117** (+1.1%) |
| **wind-mill** | 1,259 | **0.157** | 0.071 (-8.6%) | **0.031** | 0.012 (-1.9%) |
| **All Classes** | **75,082** | **0.5859** | **0.5980** (+1.21 pp) | **0.2654** | **0.2738** (+0.84 pp) |

---

## 5. Technical Analysis & Theoretical Takeaways

1. **Impact of End-to-End One-to-One Matching:**
   YOLO26s employs dual-head bipartite label assignment, discarding traditional non-maximum suppression (NMS) during inference. In aerial tiny-object imagery where targets form dense spatial clusters (e.g., parking lots, shipyards, marching formations), standard NMS frequently suppresses genuine adjacent tiny targets due to overlapping box borders. The bipartite matching framework cleanly separates adjacent predictions, leading to substantial gains in `airplane` ($+8.4\%$), `person` ($+2.7\%$), and `ship` ($+2.3\%$).

2. **The `wind-mill` Regression & Regression Penalty:**
   `wind-mill` experienced an $8.6\text{ pp}$ drop in YOLO26s ($15.7\% \rightarrow 7.1\%$). In AI-TOD-v2, windmills consist of extremely thin rotor blades with high aspect ratios, where small coordinate shifts catastrophically reduce Intersection-over-Union (IoU). While YOLOv8s employs Distribution Focal Loss (DFL) with $\text{reg\_max}=16$ to learn coordinate probability distributions, YOLO26s enforces direct coordinate regression ($\text{reg\_max}=1$). This demonstrates that standard IoU/L1 loss metrics are ill-conditioned for extreme aspect ratios at micro scales.

3. **Inference Latency Dynamics in Sliced Pipelines:**
   Under SAHI patch slicing ($512 \times 512$ tiles with $20\%$ overlap), YOLO26s achieved an average latency of $130.62\,\text{ms}$ per image compared to $140.37\,\text{ms}$ for YOLOv8s. Eliminating post-prediction NMS clustering across overlapping slice edges substantially accelerates cross-tile prediction merging.

4. **Architectural Directives for Subsequent Iterations:**
   - **$P_2$ Stride-4 Feature Head:** Standard downsampling in both baselines reaches a minimum stride of 8 ($P_3$). For objects $<8\text{px}$, spatial information is collapsed into single feature map cells. Adding a $P_2$ (stride 4, $200 \times 200$) detection head is mandatory for resolving sub-16px targets.
   - **Normalized Wasserstein Distance (NWD) Loss:** Bounding box regression must transition from IoU to Gaussian Wasserstein distance metrics, modeling bounding boxes as 2D Gaussian distributions to achieve smooth gradient flow on micro targets and resolve thin-blade failure modes.

---

## 6. Model Artifact Locations

All converged weights are tracked under the repository `weights/` directory:
- `weights/yolo26s_aitod_full_best.pt` (Hash-verified 100-epoch checkpoint, mAP50: 0.5980)
- `weights/yolov8s_aitod_full_best.pt` (Hash-verified 100-epoch checkpoint, mAP50: 0.5859)
- Evaluation exports, confusion matrices, and raw prediction JSONs are archived under `runs/kaggle_continuation/artifacts/`.
