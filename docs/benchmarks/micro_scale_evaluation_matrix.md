# AI-TOD-v2 Consolidated Micro-Scale & SAHI Benchmark Matrix

This document reports the comprehensive empirical evaluation results ($AP_{\text{all}}, AP_{50}, AP_{vt}, AP_t, AP_s, AP_m$) across the full AI-TOD-v2 validation dataset (2,804 aerial images, 75,082 ground-truth instances) across model generations, high-resolution $P_2$ feature pyramid heads, Normalized Wasserstein Distance (NWD) loss, and SAHI sliced inference.

---

## 1. Experimental Protocol & Scale Partitioning

In accordance with official AI-TOD-v2 evaluation standards, instances are partitioned into four closed area intervals:
- **$AP_{vt}$ (Very Tiny):** $[0, 64]\,\text{px}^2$ ($\le 8\,\text{px}$ side length). Sub-pixel at stride 8 ($P_3$).
- **$AP_t$ (Tiny):** $[64, 256]\,\text{px}^2$ ($8 - 16\,\text{px}$ side length). Characteristic aerial micro-target regime (dataset median: $11.83\,\text{px}$).
- **$AP_s$ (Small):** $[256, 1024]\,\text{px}^2$ ($16 - 32\,\text{px}$ side length).
- **$AP_m$ (Medium):** $[1024, 10^{10}]\,\text{px}^2$ ($> 32\,\text{px}$ side length). Extended structures and facilities.
- **$AP_{\text{all}}$:** Full validation benchmark mean Average Precision across all scales.

Operating protocol:
- Resolution: $800 \times 800$ native pixels.
- Evaluation confidence floor: $\text{conf} = 0.25$.
- Post-processing: Native end-to-end for YOLO26s bipartite head; class-aware NMS ($\text{IoU}=0.70$) for YOLOv8s.
- Slicing parameters (SAHI): $512 \times 512$ tile size, $0.20$ overlap ratio ($102\,\text{px}$).
- Evaluator: Standardized `COCOeval` with closed boundary intervals.

---

## 2. Milestone 2: Micro-Scale Benchmark Matrix (Full 2,804 Validation Images)

| Model Architecture | Head & Loss Specialization | Inference Post-Processing | Eval Latency (s) | $AP_{\text{all}}$ | $AP_{50}$ | $\mathbf{AP_{vt}}$ (2–8px) | $\mathbf{AP_{50,vt}}$ | $\mathbf{AP_t}$ (8–16px) | $\mathbf{AP_{50,t}}$ | $AP_s$ (16–32px) | $AP_m$ (32–64px) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8s Baseline** | $P_3, P_4, P_5$ (Stride 8, 16, 32) + CIoU/DFL | Standard Class-Aware NMS | 863.2s | 0.2689 | 0.5905 | 0.0874 | 0.2564 | 0.2507 | 0.5927 | **0.3514** | **0.4644** |
| **YOLO26s Baseline** | $P_3, P_4, P_5$ (Stride 8, 16, 32) + STAL Assign | Native End-to-End (NMS-free) | **874.6s** | **0.2713** | 0.5900 | **0.1437** | **0.3998** | **0.2558** | **0.5885** | 0.3428 | 0.4499 |
| **YOLOv8s-P2 + NWD** | $P_2, P_3, P_4, P_5$ (Stride 4) + NWD ($\alpha=0.50$) | Standard Class-Aware NMS | 1079.7s | 0.2675 | 0.5892 | **0.1117** | **0.3276** | 0.2537 | 0.5858 | 0.3379 | 0.4445 |
| **YOLO26s-P2 + NWD** | $P_2, P_3, P_4, P_5$ (Stride 4) + NWD ($\alpha=0.50$) | Native End-to-End (NMS-free) | 1290.6s | 0.2667 | 0.5769 | **0.1355** | **0.3792** | 0.2554 | 0.5827 | 0.3333 | 0.4533 |

---

## 3. Milestone 3: Full Validation SAHI Sliced Inference Benchmark (2,804 Images)

| Model Architecture | Slicing Resolution | Overlap Ratio | Images Evaluated | Total Detections | Mean Latency per Image | Total Wall Time | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26s Baseline** | $512 \times 512$ | 0.20 | 2,804 | 85,518 | **159.37 ms** | **446.88s** | **6.27 FPS** (Fastest) |
| **YOLOv8s Baseline** | $512 \times 512$ | 0.20 | 2,804 | 87,174 | 173.21 ms | 485.69s | 5.77 FPS |
| **YOLO26s-P2 + NWD** | $512 \times 512$ | 0.20 | 2,804 | 84,148 | 190.56 ms | 534.36s | 5.25 FPS |
| **YOLOv8s-P2 + NWD** | $512 \times 512$ | 0.20 | 2,804 | 86,645 | 228.03 ms | 639.41s | 4.39 FPS |

---

## 4. Empirical Findings & Architectural Insights

### 4.1 Sub-8-Pixel Recovery Dynamics ($AP_{vt}$)
- **YOLO26s Architectural Superiority on Micro Targets:**
  YOLO26s achieves **14.37% $AP_{vt}$** and **39.98% $AP_{50,vt}$**, vastly outperforming YOLOv8s baseline (**8.74% $AP_{vt}$**, **25.64% $AP_{50,vt}$**)—an absolute increase of **+5.63 percentage points (+64.4% relative gain)** on sub-8px objects without requiring extra heads.
- **P2 Stride-4 Head & NWD Loss on YOLOv8s:**
  Integrating the high-resolution $P_2$ feature pyramid head and Gaussian Wasserstein distance loss yields an absolute increase of **+2.43 percentage points (+27.8% relative gain)** in $AP_{vt}$ ($8.74\% \to 11.17\%$) and **+7.12 percentage points** in $AP_{50,vt}$ ($25.64\% \to 32.76\%$).
- **NWD Distribution Modeling:**
  By modeling bounding boxes as 2D Gaussian distributions, NWD guarantees scale-smooth, non-zero gradient signals when ground truth and predicted boxes have zero pixel overlap, preventing gradient vanishing on microscopic targets.

### 4.2 SAHI Throughput & Latency Dynamics
- **Elimination of NMS Bottlenecks:**
  YOLO26s baseline achieved the fastest SAHI slicing throughput (**6.27 FPS / 159.37 ms**), running ~8% faster than YOLOv8s baseline (5.77 FPS) and ~30% faster than YOLOv8s-P2+NWD (4.39 FPS).
- **Patch Merging Efficiency:**
  Because YOLO26s features a bipartite end-to-end head without post-prediction greedy NMS clustering, merging bounding box predictions across patch boundaries incurs substantially lower computation and coordinate jitter.

---

## 5. Artifact Ledger

All raw prediction files, JSON matrices, and precision-recall curves are archived locally in `evaluation_results_consolidated/output/`:
- `micro_scale_benchmark_matrix.json`: Full nested dictionary of AP/AR metrics per model and category.
- `sahi_full_benchmark_matrix.json`: Sliced inference latencies and detection tallies.
- Precision-recall curves per model: `evaluation_results_consolidated/output/micro_eval/{model}/evaluation/curves.npz`.
- Standardized COCO prediction streams: `evaluation_results_consolidated/output/micro_eval/{model}/prediction_export/predictions.json`.
