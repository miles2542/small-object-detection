# Runtime, Throughput, and VRAM/RAM Benchmark Report

**Dataset Baseline:** AI-TOD-v2 ($800 \times 800$, tiny objects $\le 16\times 16$ px) & VisDrone2019-DET  
**Models Evaluated:** YOLOv8s (`yolov8s.pt`, 11.17M params) vs. YOLO26s (`yolo26s.pt`, 10.01M params)  
**Host Architecture:** 12th Gen Intel Core i5-12600K (10 cores, 16 threads, 32GB RAM), AMD Radeon RX 6700 XT 12GB  
**Software Stack:** Windows 11 64-bit, Python 3.11, PyTorch 2.14.1+cpu, Ultralytics 8.4.184, SAHI 0.12.8

---

## 1. Executive Summary & Verification Findings

1. **Model Catalog & Naming:**
   - **YOLOv8s:** Default checkpoint `yolov8s.pt` (11,166,560 parameters, standard `Detect` head with DFL layer).
   - **YOLO26s:** Ultralytics (v8.4.184) registers and ships YOLO26 natively under `yolo26s.pt` and `yolo26s.yaml`. It is **not** named `yolov26s`. YOLO26s features 10,009,784 parameters with an NMS-free dual head (`one2one_cv2`, `one2one_cv3`, `Identity()` DFL).
2. **Acceleration & Hardware Compatibility:**
   - PyTorch was tested with `torch-directml` for AMD Radeon RX 6700 XT on Windows 11. However, the pre-built `torch-directml` wheels on Windows encounter DLL procedure entry point errors (`torch_directml_native` incompatible with current C++ runtime / torch ABI). Therefore, local inference operates cleanly on CPU.
   - On the Intel Core i5-12600K CPU, YOLO26s demonstrates superior efficiency over YOLOv8s at native AI-TOD resolution ($800 \times 800$) and higher resolutions, achieving **381.7 ms** (2.62 FPS) vs. **420.9 ms** (2.38 FPS) for YOLOv8s.
3. **Real Aerial Imagery Throughput:**
   - On real aerial drone imagery ($800 \times 800$), YOLO26s achieves **248.6 ms** (4.02 FPS) compared to **282.8 ms** (3.54 FPS) for YOLOv8s, with a low resident memory footprint of under **700 MB**.
4. **SAHI Sliced Inference Trade-off:**
   - SAHI tiling ($512 \times 512$ slices with 0.2 overlap) on $800 \times 800$ frames generates 4 slices per image, resulting in **1194.7 ms** latency (0.84 FPS) for YOLO26s vs. **1426.5 ms** (0.70 FPS) for YOLOv8s.
5. **Training Step Verification:**
   - Dummy forward and backward gradient passes run cleanly on both models without memory errors or gradient mismatches ($800 \times 800$, batch size 2).

---

## 2. Empirical Benchmark Measurements

### A. Resolution Sweep (Batch Size = 1, Default Threads)

| Model | Input Resolution | Cold Start (ms) | Mean Latency (ms) | P50 (ms) | P95 (ms) | FPS | Peak RAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8s** | $640 \times 640$ | 978.0 | **88.1** | 75.0 | 143.9 | **11.35** | 539.9 |
| **YOLO26s** | $640 \times 640$ | 372.8 | 238.9 | 243.6 | 254.5 | 4.19 | 580.1 |
| **YOLOv8s** | $800 \times 800$ (AI-TOD) | 449.0 | 420.9 | 418.6 | 443.7 | 2.38 | 606.8 |
| **YOLO26s** | $800 \times 800$ (AI-TOD) | 401.9 | **381.7** | **375.5** | **418.8** | **2.62** | 636.5 |
| **YOLOv8s** | $1024 \times 1024$ | 698.6 | 690.2 | 678.9 | 757.4 | 1.45 | 649.4 |
| **YOLO26s** | $1024 \times 1024$ | 600.1 | **620.4** | **602.3** | **715.7** | **1.61** | 702.2 |
| **YOLOv8s** | $1280 \times 1280$ | 1056.8 | 1057.7 | 1054.7 | 1108.5 | 0.95 | 708.4 |
| **YOLO26s** | $1280 \times 1280$ | 1042.4 | **949.4** | **949.4** | **1006.0** | **1.05** | 751.9 |

> **Key Takeaway:** At standard $640 \times 640$, YOLOv8s is faster on CPU due to shallow convolution blocks, but at AI-TOD native $800 \times 800$ and above, YOLO26s scales better, beating YOLOv8s in both mean latency and P95 latency.

---

### B. Batch Size Scaling ($800 \times 800$ Native AI-TOD Resolution)

| Model | Batch Size | Total Step (ms) | Latency / Img (ms) | Throughput (FPS) | Peak RAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8s** | 1 | 454.3 | 454.3 | 2.20 | 642.0 |
| **YOLO26s** | 1 | **386.3** | **386.3** | **2.59** | 652.3 |
| **YOLOv8s** | 2 | 872.4 | 436.2 | 2.29 | 676.4 |
| **YOLO26s** | 2 | **759.5** | **379.7** | **2.63** | 725.7 |
| **YOLOv8s** | 4 | 1726.9 | 431.7 | 2.32 | 784.8 |
| **YOLO26s** | 4 | **1466.0** | **366.5** | **2.73** | 836.2 |
| **YOLOv8s** | 8 | 3416.5 | 427.1 | 2.34 | 1064.9 |
| **YOLO26s** | 8 | **2877.5** | **359.7** | **2.78** | 1187.4 |
| **YOLOv8s** | 16 | 6495.1 | 405.9 | 2.46 | 1614.4 |
| **YOLO26s** | 16 | **5505.2** | **344.1** | **2.91** | 1732.2 |

---

### C. CPU Thread Scaling ($800 \times 800$, Batch Size = 1)

| Model | Threads | Mean Latency (ms) | FPS | Scaling Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv8s** | 1 | 1162.5 | 0.86 | 1.00x (baseline) |
| **YOLO26s** | 1 | 972.7 | 1.03 | 1.00x (baseline) |
| **YOLOv8s** | 2 | 632.3 | 1.58 | 1.84x |
| **YOLO26s** | 2 | 549.0 | 1.82 | 1.77x |
| **YOLOv8s** | 4 | 471.3 | 2.12 | 2.47x |
| **YOLO26s** | 4 | 425.0 | 2.35 | 2.29x |
| **YOLOv8s** | 8 | **418.3** | **2.39** | 2.78x (peak) |
| **YOLO26s** | 8 | 393.4 | 2.54 | 2.47x |
| **YOLOv8s** | 16 | 460.8 | 2.17 | 2.52x (thread contention) |
| **YOLO26s** | 16 | **369.8** | **2.70** | **2.63x** |

> **Key Takeaway:** 8 threads matches the physical performance cores of the i5-12600K optimally. Allocating beyond 8 threads causes thread contention overhead for YOLOv8s, whereas YOLO26s continues to gain throughput at 16 threads.

---

### D. Real Aerial Drone Imagery ($800 \times 800$, VisDrone Sample)

| Model | Sample Set | Mean Latency (ms) | P50 (ms) | P95 (ms) | Throughput (FPS) | Peak RAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8s** | 20 real drone frames | 282.8 | 277.3 | 321.3 | 3.54 FPS | 678.4 |
| **YOLO26s** | 20 real drone frames | **248.6** | **248.3** | **280.4** | **4.02 FPS** | 690.5 |

---

### E. SAHI Tiling vs. Full-Frame Inference

| Model | Mode | Input Resolution | Slices | Latency (ms) | FPS | Peak RAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8s** | Full-frame | $800 \times 800$ | 1 | 420.9 | 2.38 | 606.8 |
| **YOLOv8s** | SAHI ($512\times 512$, ov 0.2) | $800 \times 800$ | 4 | 1426.5 | 0.70 | 754.6 |
| **YOLO26s** | Full-frame | $800 \times 800$ | 1 | **381.7** | **2.62** | 636.5 |
| **YOLO26s** | SAHI ($512\times 512$, ov 0.2) | $800 \times 800$ | 4 | **1194.7** | **0.84** | 734.2 |
| **YOLOv8s** | Full-frame | $1280 \times 1280$ | 1 | 1057.7 | 0.95 | 708.4 |
| **YOLOv8s** | SAHI ($512\times 512$, ov 0.2) | $1280 \times 1280$ | 9 | 2810.2 | 0.36 | 735.6 |
| **YOLO26s** | Full-frame | $1280 \times 1280$ | 1 | **949.4** | **1.05** | 751.9 |
| **YOLO26s** | SAHI ($512\times 512$, ov 0.2) | $1280 \times 1280$ | 9 | **2479.8** | **0.40** | 742.9 |

---

### F. Dummy Training Step Verification

| Model | Batch Size | Input Resolution | Mean Step Latency (ms) | Peak RAM (MB) | Gradient Pass Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8s** | 2 | $800 \times 800$ | 4348.4 | 1142.0 | **PASS** (Zero errors) |
| **YOLO26s** | 2 | $800 \times 800$ | 4390.3 | 1273.2 | **PASS** (Zero errors) |

---

## 3. Recommendations & Architectural Design Decisions

1. **Model Selection & Catalog Registration:**
   - Use `yolov8s.pt` and `yolo26s.pt` as the exact checkpoint strings in YAML configs and CLI invocations.
2. **Batch & Thread Configuration for CPU / Local Dev:**
   - Set `workers: 4` and thread count to 8 on the i5-12600K for maximum throughput efficiency without thread contention.
   - For validation sweeps, batch size 8 or 16 maximizes CPU throughput (344 ms/image vs 386 ms/image).
3. **Inference Strategy & Micro-Scale Trade-off:**
   - Full-frame $800 \times 800$ inference runs at $\sim 4$ FPS on real drone imagery with low memory footprint (<700 MB).
   - SAHI tiling increases inference latency by $\sim 3.5\times$ (4 slices on $800 \times 800$, 9 slices on $1280 \times 1280$), reducing frame rates to $\sim 0.8$ FPS.
4. **Execution Infrastructure (Local CPU vs. Cloud GPU):**
   - Local workstations are dedicated to data ingestion, EDA, format conversions, unit tests, and lightweight CPU inference.
   - Resource-intensive model training (full 100-epoch runs for YOLOv8s and YOLO26s baselines and the factorial ablation matrix) is routed to cloud environments with dual NVIDIA T4 GPUs (Kaggle/Colab) and teammate CUDA hardware (RTX 3060). Remote training sweeps will be orchestrated via programmatic Kaggle CLI workflows (`kaggle kernels push`/`status`/`output`).
5. **Factorial Ablation Matrix Integration:**
   - The runtime profiling confirms that testing higher input resolutions ($1280 \times 1280$) and integrating high-resolution P2 stride-4 heads will introduce compute overheads that are best absorbed by cloud GPU infrastructure rather than CPU local machines.
