# AI-TOD-v2 Official Micro-Scale Evaluation Benchmark Matrix

This document reports the empirical micro-scale evaluation metrics ($AP_{\text{all}}, AP_{vt}, AP_t, AP_s, AP_m$) across the full AI-TOD-v2 validation dataset (2,804 aerial images, 75,082 ground-truth instances) across model generations, high-resolution $P_2$ feature pyramid heads, and Normalized Wasserstein Distance (NWD) loss.

---

## 1. Evaluation Protocol & Scale Partitioning

In accordance with official AI-TOD-v2 evaluation standards, instances are partitioned into four closed area intervals:
- **$AP_{vt}$ (Very Tiny):** $[0, 64]\,\text{px}^2$ ($\le 8\,\text{px}$ side length). Sub-pixel at stride 8 ($P_3$).
- **$AP_t$ (Tiny):** $[64, 256]\,\text{px}^2$ ($8 - 16\,\text{px}$ side length). Characteristic aerial micro-target regime.
- **$AP_s$ (Small):** $[256, 1024]\,\text{px}^2$ ($16 - 32\,\text{px}$ side length).
- **$AP_m$ (Medium):** $[1024, 10^{10}]\,\text{px}^2$ ($> 32\,\text{px}$ side length). Extended structures and facilities.
- **$AP_{\text{all}}$:** Full validation benchmark mean Average Precision across all scales.

Operating protocol:
- Resolution: $800 \times 800$ native pixels.
- Score floor: $\text{conf} = 0.25$.
- NMS / Deduplication: Native end-to-end for YOLO26s bipartite head; class-aware NMS ($\text{IoU}=0.70$) for YOLOv8s.
- Evaluator: Standardized `COCOeval` with closed boundary intervals.

---

## 2. Benchmark Comparison Matrix (2,804 Images, 75,082 Instances)

| Model Architecture | Head & Loss Specialization | Inference Post-Processing | Eval Latency (s) | $AP_{\text{all}}$ | $\mathbf{AP_{vt}}$ (2–8px) | $\mathbf{AP_t}$ (8–16px) | $AP_s$ (16–32px) | $AP_m$ (32–64px) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8s Baseline** | $P_3, P_4, P_5$ (Stride 8, 16, 32) + CIoU/DFL | Standard Class-Aware NMS | 1055.0s | 0.2689 | **0.0874** | 0.2507 | 0.3514 | **0.4644** |
| **YOLO26s Baseline** | $P_3, P_4, P_5$ (Stride 8, 16, 32) + STAL Assign | Native End-to-End (NMS-free) | **922.8s** | **0.2713** | **0.1437** | **0.2558** | 0.3428 | 0.4499 |
| **YOLOv8s-P2 + NWD** | $P_2, P_3, P_4, P_5$ (Stride 4) + NWD ($\alpha=0.50$) | Standard Class-Aware NMS | 1102.7s | 0.2675 | **0.1117** | 0.2537 | 0.3379 | 0.4445 |
| **YOLO26s-P2 + NWD** | $P_2, P_3, P_4, P_5$ (Stride 4) + NWD ($\alpha=0.50$) | Native End-to-End (NMS-free) | *In Progress* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

---

## 3. Empirical Findings & Architectural Insights

### 3.1 YOLO26s Small-Target-Aware Label Assignment (STAL)
- **Extreme Scale Jump on $AP_{vt}$:**
  Without any architectural modifications or extra heads, YOLO26s improves $AP_{vt}$ from **8.74%** to **14.37%**—an absolute increase of **+5.63 percentage points (+64.4% relative gain)** over YOLOv8s.
- **Bipartite Label Assignment Advantage:**
  In dense micro-scale clusters, YOLOv8s TaskAlignedAssigner forces greedy NMS during inference, which discards valid adjacent boxes. YOLO26s's end-to-end one-to-one matching resolves clustering ambiguities without suppression artifacts.

### 3.2 High-Resolution $P_2$ Stride-4 Head & NWD Loss on YOLOv8s
- **Recovery of Sub-Pixel Targets:**
  Integrating the stride-4 $P_2$ feature pyramid head and Gaussian Wasserstein distance loss yields an absolute increase of **+2.43 percentage points (+27.8% relative gain)** in $AP_{vt}$ ($8.74\% \rightarrow 11.17\%$).
- **Geometric Invariance of NWD:**
  By modeling bounding boxes as 2D Gaussian distributions, NWD guarantees non-zero, scale-smooth gradient signals even when ground-truth and prediction boxes have zero intersection—preventing gradient collapse on objects $<8\text{px}$.
