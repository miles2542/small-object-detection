# Evaluation environment and dataset preparation

**Date:** 2026-11-01 (Asia/Saigon)

**Source revision:** `24825f6`

**Branch:** `feat/m2-evaluation-coco`

**Status:** **Preparation complete.** Environment, credentials, source review, real-data preparation and final checks passed. **Historical checkpoint: before Phase 1.** See the report index for the current completed implementation. Final trained-model accuracy benchmarking awaits Member 1's checkpoint handoff.

## 1. Objective

Establish a usable local workspace, understand Member 2's complete assignment, and identify conditions that would invalidate later benchmark results. The [working plan](../member2_working_plan.md) defines the next five phases and their completion gates.

## 2. Materials and method

Read all three assignment PDF pages, with text extraction and rendered-page inspection; examined both supplied screenshots. Reviewed `README.md`, all existing `docs/` reports, `pyproject.toml`, `uv.lock`, dataset/model configuration, CLI, data download/conversion/subset/EDA modules, training and Kaggle orchestration, checked-in Kaggle runner/metadata, plotting/UI utilities and GitHub CI. `src/eval/` and `src/infer/` currently contain package initialization only.

The PDF requires more than the screenshots' evaluator outline: **operating-point precision/recall**, **class-by-size analysis**, **quantified degradation** and **representative missed/localization failures** are required deliverables. Its other members' numbering differs from the quickstart; Member 2's size-evaluation role agrees across sources.

The user authorized preparation now and implementation beginning with Phase 1 in a subsequent turn. Document-embedded directions to push kernels, train or submit a PR were treated as project guidance, not immediate execution requests.

## 3. Environment results

| Check/action | Observed result | Interpretation |
| :--- | :--- | :--- |
| Initial Git state | Clean `main`, revision `24825f6` | No pre-existing tracked changes to overwrite |
| Feature branch | Created `feat/m2-evaluation-coco` | Work is isolated from `main` |
| Initial Python environment | `.venv` referenced unavailable CPython 3.11.13 | The existing directory was unusable; it did not prove dependencies were ready |
| Environment rebuild | `uv sync --locked` succeeded with **CPython 3.12.14**, 88 installed packages | Used the existing lockfile without changing dependency definitions |
| Repeat sync | `uv sync --locked --offline` checked 88 packages successfully | Environment matches the lockfile |
| Runtime | **PyTorch 2.14.1+cpu**, **Ultralytics 8.4.184**; CUDA unavailable | Use CPU for development/evaluation smoke checks; heavy training needs a separately verified GPU environment |
| CLI | Activated `sod --help`, data download/prepare help succeeded | Project entry point works |
| Evaluation command | `sod eval --help` has no implemented subcommands | `sod eval run` remains Member 2 implementation work |
| Figure utilities | Matplotlib/Seaborn and `set_publication_style()` import/configure successfully | Reuse existing project typography, palette, headers and save helpers |
| Evaluator dependency | `pycocotools` not installed or declared | Add it explicitly during evaluator implementation, not as an untracked ad hoc dependency |
| Kaggle credential | Saved in user-profile `.kaggle/access_token`, outside repository | No credential is written into project code, docs or Git |
| Kaggle connection | `kaggle competitions list --csv` exited 0 with returned rows | Read-only connectivity/authentication smoke check passed; no cloud job was launched |
| Trained baseline artifacts | No local `runs/**/best.pt`; Member 1 reports full runs pending | Accuracy benchmarks await genuine checkpoints; harness implementation can proceed with fixtures/pretrained smoke tests |

`uv` was not initially on PATH. Its executable is available inside **`.venv/Scripts/`** after bootstrap, so activate this environment before using `uv` directly. Bootstrap packages and temporary review artifacts are removed after preparation; this executable is retained as a required environment tool. The base Python comes from the desktop bundled runtime; if that runtime is moved, rebuild `.venv` using another compatible CPython and the same lockfile.

### Commands for a new PowerShell terminal

Run from the repository root:

```powershell
. .venv/Scripts/Activate.ps1
uv sync --locked
sod --help
kaggle competitions list --csv
```

Activation applies to the terminal executing it; it cannot activate a separate terminal already open in your editor. If PowerShell activation is restricted, use the existing `sod.cmd` wrapper or `.venv/Scripts/python.exe -m src.cli --help`. No machine-wide execution policy was changed.

## 4. Annotation results and data preparation cautions

Downloaded the annotation archive using the repository's configured Google Drive ID. These counts are **observed from the downloaded JSONs**, not copied from the report:

| Property | Existing EDA report | Downloaded v2 annotations | Consequence |
| :--- | :--- | :--- | :--- |
| Train images / annotations | 11,214 / 301,534 | 11,214 / 301,534 | Train headline counts agree |
| Validation images / annotations | 2,804 / 70,424 | **2,804 / 75,091** | Recompute validation support before evaluation |
| Validation zero-dimension boxes | 0 | **9** | Explicit invalid-GT policy is needed |
| Combined train+validation annotations | Split rows sum to 371,958, while trainval row says 376,625 | **376,625** | Current v2 split counts reconcile with the trainval headline |
| Combined zero-dimension boxes | Split rows sum to 40, trainval says 49 | **49** (40 train + 9 validation) | Current v2 invalid counts reconcile too |

| File | SHA-256 |
| :--- | :--- |
| `data/ai-tod-v2/annotations/aitodv2_train.json` | `ed7b37a1187b496b96943fa46c15aab39656d59eaf501192d85178354b637b2e` |
| `data/ai-tod-v2/annotations/aitod_val.json` | `d0439ba687db66bb38584a3398a906f15c47e055cdd84bb9060496a91f30d7cf` |

The archive contains `aitod_train_v2.json` and `aitod_val_v2.json`; the existing downloader creates the canonical aliases above. Existing report discrepancies suggest inconsistent source annotations, but the historical file version cannot be proven from this clone alone. Leave original files/report unchanged and verify all current protocol inputs in Phase 1.

**Observed onboarding traps:**

| Issue | Code evidence | Preparation approach |
| :--- | :--- | :--- |
| Documented full split uses sizes `0` | `select_stratified_subset(..., target_size=0)` returned **zero selected images** in an in-memory check | Use actual positive image counts for full preparation; do not run the documented zero-size command |
| Placeholders enabled by default | CLI/converter can generate blank images when sources are absent | Always pass **`--no-placeholders`** and verify real copied-image counts |
| Missing sources can still be skipped without placeholders | Converter writes labels without ensuring every source image exists | Independently compare image/label counts against selected IDs |
| Archive nesting | Downloaded ZIP local headers start with `train/` and `val/`, but extraction destination already includes the split | Normalize the resulting nested directories before conversion, or explicitly supply the actual source directories |
| CLI split list | Typer exposes repeatable `--splits`; README uses `--splits train val` | Use `--splits train --splits val`, or the existing downloader module which accepts both values after one flag |
| Mini sampling | Greedy approximate class/scale matching with rare-class boost | Validate distributions; do not promise exact preservation |
| Bounding-box policy | Converter rejects nonpositive dimensions and clips coordinates; it does not universally drop all sub-1 px boxes | Document actual policy instead of adopting the EDA prose literally |
| EDA trainval fallback | Missing combined JSON causes a copy of train statistics labeled as derived trainval | Do not use that fallback to establish combined statistics |

Real image archive downloads were started through the existing downloader, with train/validation handled concurrently. Validation is now complete: **2,804 PNGs** match the annotation filenames and dimensions and pass PNG integrity checks. The nested archive image directory was normalized to `images/val/`. The archive's original text labels remain preserved under `images/val/val/labels/`; conversion uses the **v2 COCO JSON**, not those supplied text labels. Automatic review rejected a proposed recursive cleanup that would have removed that metadata, so the safer image-only move was used successfully.

| Prepared validation dataset | Images | YOLO label files | Valid boxes | Zero-dimension boxes dropped | Placeholders |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Mini, seed 42 | 200 | 200 | 5,054 | 0 | **0** |
| Full | 2,804 | 2,804 | 75,082 | 9 | **0** |

Training download and extraction are complete. All **11,214 training PNGs** match the annotation filenames and **800 × 800** dimensions; **33 sampled training PNGs** also passed PNG integrity checks. ZIP extraction completed without CRC errors. Training PNGs were moved to `images/train/`; original archive labels remain under `images/train/train/labels/`.

The **1,000-image training mini split** contains **24,877 valid converted boxes**, with **4 zero-dimension boxes dropped** and **zero placeholders**. Training images in the prepared splits are **hard links** to the canonical raw images on the same volume, so they share the underlying image data; treat those source/split images as immutable. Validation images were copied by the existing converter. The mini validation IDs and training IDs/sampling settings are preserved in ignored `data/ai-tod-v2/preparation_summary.json`.

Full training conversion is complete: **11,214 image/label pairs**, **301,494 valid boxes**, **40 zero-dimension boxes dropped**, **zero out-of-bounds boxes dropped** and **zero placeholders**. Both existing tracked YAMLs match the converter's generated configuration semantically; no machine-specific YAML paths were committed.

**Final checks passed:** Ultralytics `check_det_dataset(..., autodownload=False)` resolves both existing YAMLs to the intended prepared paths and eight classes. All four image/label filename sets match their expected counts. **Every written YOLO label row** has five fields, class ID 0–7, finite normalized coordinates within [0,1], and positive width/height. Training hard-link identity was checked for each prepared dataset. The downloader exited successfully; `data/_downloads/` is empty. Temporary PDF renders, bootstrap packages/cache, logs and inspection files were removed; the required `uv` executable remains inside `.venv/Scripts/`.

### Prepared dataset inventory

| Split | Training images | Validation images | Training valid boxes | Validation valid boxes | Placeholders |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Mini**, seed 42 | **1,000** | **200** | **24,877** | **5,054** | **0** |
| **Full** | **11,214** | **2,804** | **301,494** | **75,082** | **0** |

Existing configurations are `configs/aitod_mini.yaml` and `configs/aitod_full.yaml`. The original source JSONs are unchanged. Image/label conversion used the existing converter functions; hard links were populated before training conversion, so its `images_copied` counter is zero for training and the summary records `images_hardlinked_before_conversion` separately.

For an equivalent regeneration of YOLO labels with the current canonical raw image layout, use:

```powershell
# Development split; keep placeholders disabled
sod data prepare --train-size 1000 --val-size 200 --seed 42 --no-placeholders

# Full split: use the actual counts, not the documented zero-size convention
sod data prepare --train-size 11214 --val-size 2804 --seed 42 --no-placeholders --output-dir data/ai-tod-v2/splits/full --config-yaml configs/aitod_full.yaml
```

The downloader was invoked as `python -m src.data.download --dataset aitod-v2 --splits train val`, with validation downloaded concurrently through `download_aitod(..., splits=['val'])`. For the Typer CLI, spell multiple splits as `sod data download --dataset aitod-v2 --splits train --splits val`. On a fresh clone, inspect and normalize archive nesting before conversion; the prepared workspace already has the canonical paths.

## 5. Evaluation design findings

Use **original COCO annotations and IDs** as the evaluation surface; YOLO labels are a training surface. Generic COCO-pretrained weights have a different class vocabulary and cannot establish this project's eight-class baseline accuracy. The PDF requires checkpoints from baseline training, not merely a downloadable model filename.

The requested four bins use area bounds **[4,64], [64,256], [256,1024], [1024,4096] px²**. The [upstream evaluator source](https://github.com/jwwangchn/cocoapi-aitod/blob/master/aitodpycocotools/aitodpycocotools/cocoeval.py) currently uses different outer bounds and a 1,500-detection maximum. Shared endpoints are included in both adjacent evaluation ranges. **Assignment-specific metrics and upstream-compatible metrics need explicit names and metadata.** Simply replacing `areaRng` while using the default summary function is insufficient.

The AP export confidence floor must remain low enough to preserve PR curves. The operating-point threshold for precision/recall is a different setting. Missing class/size support must be **N/A**, not false zero performance. Full evaluator correctness and matching tests belong to Phase 3.

## 6. Kaggle and benchmark limitations

The [official Kaggle authentication reference](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md) supports a user-profile `access_token`. The successful API read does not establish available GPU quota, accelerator availability or a complete training pipeline.

The repository's Kaggle defaults belong to **`miles2542`**, whereas this user's account is **`harvey510`**. The checked-in metadata slug also differs from the orchestration default. Do not push or fetch a default kernel assuming it is the user's job.

The checked-in `kaggle/kernels/aitod_train/runner.py` **synthesizes constant-color images and random labels**. It is a throughput profiler, not evidence of AI-TOD detection quality. The scaffold generator is different and fetches annotations without completing real imagery preparation. Neither route should be launched unchanged to obtain scientific baseline checkpoints. No kernel was scaffolded, pushed or trained during this preparation.

The existing throughput projections refer to a **1,000-image** workload. They are not timing evidence for an 11,214-image training epoch or this user's hardware. The presence of two remote GPUs also does not prove a run used both; the generated runner selects device `0`.

**Updated teammate handoff:** Member 1 reports that full baseline training at `miles2542/ai-tod-baseline-training` is queued/in progress, with final checkpoints to be shared later. We have not independently verified that run's status or contents. The absence of weights is expected because `.pt` and `runs/` are ignored. Follow the teammate's instruction to implement and verify the harness now using analytic fixtures and optionally pretrained predictions labeled **smoke-only**. Such predictions require explicit class-name mapping; they are not comparable eight-class AI-TOD baseline accuracy results. Member 1 plans to rerun evaluation after the trained checkpoints arrive.

## 7. Visualizations and practical implications

**Visualization status:** No model inference, AP, PR curves or error gallery has been generated during onboarding. Such figures require genuine trained checkpoints, validated predictions and the frozen protocol. A setup discrepancy table is sufficient evidence here; inventing an AP chart would misrepresent progress.

Future figures will call the existing plotting helpers, use **300 dpi**, existing scale-bin colors and font/header conventions, and be visually inspected after saving. See the working plan for required Phase 5 figures and interpretation.

**Verified annotation version + real imagery + original IDs** $\implies$ **a defensible basis for later size-based comparison**. **Blank synthetic imagery or mismatched classes** $\implies$ **results that cannot answer the aerial tiny-object question**, even if training/evaluation code executes.

## 8. Outputs and next decision

Created only the requested plan and report folder/index/report as tracked additions. Real datasets, the environment and preparation provenance remain in ignored locations. Document links and absence of credential text were checked. Main evaluation code has not been implemented. Existing team documentation, source code, configs, dependency definitions and lockfile remain unchanged. No commit, push, PR or cloud training job was created.

**Next:** review [Phase 1](../member2_working_plan.md#phase-1--audit-inputs-and-freeze-the-evaluation-protocol), then confirm its implementation in the next turn. Important decisions to retain are annotation version, assignment/upstream protocol distinction, detection caps and checkpoint provenance. Any missing trained checkpoints must be supplied or obtained through an accessible artifact handoff before genuine baseline benchmarking.


**Subsequent milestone:** [Phase 1 input audit and protocol](01_protocol_and_inputs.md) is complete; its findings supersede the provisional evaluation choices in this preparation record.
