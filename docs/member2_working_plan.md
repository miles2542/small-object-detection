# Size-based evaluation working plan

**Owner:** Member 2 (Thảo) | **Prepared:** 2026-11-01 (Asia/Saigon)

**Branch:** `feat/m2-evaluation-coco` | **Current stage:** Phases 1-5 implementation and diagnostic handoff complete; genuine trained full-validation benchmarks and conclusions remain pending.

## 1. Purpose and source interpretation

Answer: **How does detection quality change with object size, and how severe is degradation for very tiny/tiny objects?** Build one reproducible evaluator that other members can reuse for baseline, P2, SAHI, and NWD predictions.

| Material | What it establishes | How this plan uses it |
| :--- | :--- | :--- |
| Parent-folder `_task assignment.pdf`, Member 2 row, page 1 | AP by size; precision/recall; class analysis; performance degradation; representative missed/localization failures | Defines the complete Member 2 deliverable scope |
| Two supplied screenshots | COCO JSON export, custom evaluator, `sod eval run`, two baseline runs, feature branch and eventual PR | Provides an implementation outline; does not replace the PDF's additional analysis requirements |
| `README.md`, `team_quickstart.md` | Environment, data, training and collaboration conventions | Reuse existing infrastructure after checking actual behavior |
| `dataset_analysis.md`, existing code/configuration | Tiny-object motivation and current integration interfaces | Treat numerical and causal claims as inputs to audit, not independently verified measurements |

**Execution checkpoints:** preparation and Phases 1-5 implementation are complete. Phase 4/5 official trained benchmarks and final scientific conclusions remain pending external artifacts. Review the Phase 5 handoff before publication or further experiments. Training/publication are separate milestones; no teammate messages have been sent.

The PDF assigns inference to Member 3, architecture/NWD to Member 4, and TIDE to Member 5; the quickstart distributes those roles differently. Member 2 is consistently responsible for size-based evaluation. Refer to collaborators by **function** until the team resolves the other numbering.

## 2. Project fundamentals

**Dual surface** means COCO source annotations → YOLO labels for training → predictions mapped back to original COCO image/category IDs for evaluation. Keep original annotations immutable. **Scale** is $s=\sqrt{w h}$ in original-image pixels; **area** is in px². Neither means the maximum box side.

Dataset concentration motivates size-specific metrics, but the repository's aggregate statistics need rechecking: the listed train and validation annotation counts sum to **371,958**, not the reported trainval **376,625**. The downloaded v2 archive actually contains **75,091 validation annotations**, including **9 zero-dimension boxes**, rather than the EDA report's 70,424 and zero. Its train total 301,534 plus validation total 75,091 does equal 376,625. The downloader correctly aliases `aitod_val_v2.json` to `aitod_val.json`; a filename alone does not establish the version. Dense vehicles can dominate pooled instance statistics, whereas COCO AP averages valid category slices. Report both **support counts** and **per-class metrics**.

**Small feature footprints and IoU sensitivity motivate hypotheses.** They do not prove that deeper layers have zero activation, that every tiny target receives zero assignments, or that a particular intervention must improve AP. Establish measured baseline behavior before interpreting an architectural cause.

**Member 1 handoff, received during preparation:** full baseline training at `miles2542/ai-tod-baseline-training` is reported queued/in progress; converged checkpoints are not published yet and will be shared later. This is a teammate-provided status, not a verified local training result. **Do not wait for convergence to implement Phases 1–3 or the Phase 4 CLI.** Use analytic COCO fixtures and, when useful, clearly labeled pretrained smoke predictions. COCO-pretrained classes differ from AI-TOD; any smoke adapter must declare an explicit class-name mapping and reject unsupported/ambiguous classes. Never map COCO class indices directly to AI-TOD IDs. Trained checkpoint benchmarking remains a pending handoff, to be rerun by Member 1 or this workspace when genuine artifacts arrive.

### Assignment protocol and compatibility questions

| Metric group | Requested area interval (px²) | Corresponding scale (px) |
| :--- | :--- | :--- |
| All | $[0, 10^{10}]$ | All valid sizes |
| Very tiny, $AP_{vt}$ | $[2^2,8^2]=[4,64]$ | 2–8 |
| Tiny, $AP_t$ | $[8^2,16^2]=[64,256]$ | 8–16 |
| Small, $AP_s$ | $[16^2,32^2]=[256,1024]$ | 16–32 |
| Medium, $AP_m$ | $[32^2,64^2]=[1024,4096]$ | 32–64 |

**Boundary caution:** COCOeval excludes areas strictly below/above each bound, so shared endpoints belong to both adjacent evaluation intervals. EDA helpers instead assign each object to one interval using upper-inclusive scale thresholds. Keep those two purposes explicit; test areas exactly 4, 64, 256, 1024 and 4096, plus neighboring values. Objects below 4 or above 4096 remain in **all**, even when absent from the four requested bins.

The [upstream AI-TOD evaluator](https://github.com/jwwangchn/cocoapi-aitod/blob/master/aitodpycocotools/aitodpycocotools/cocoeval.py) inspected during onboarding uses very-tiny area starting at 0, an open-ended medium group, and `maxDets=[1,100,1500]`. Its interval endpoints are inclusive. **Do not label the screenshot's intervals as identical to this upstream version.** Phase 1 pinned reference revision `44a230ae5197cb89bf9e5e62f313cac3ad30c7af` and defined explicit, named protocol settings in `src/eval/protocol.py`. Default to the assignment intervals for the assignment deliverable; retain a clearly separate upstream-compatible comparison when needed. Detection export limits and evaluator limits must be recorded separately and checked for truncation.

## 3. Roadmap and completion gates

| Stage | Main outcome | Required completion evidence | Report |
| :--- | :--- | :--- | :--- |
| Preparation — **complete** | Read materials; establish usable environment, credentials and data readiness; identify gaps | Verified environment, Kaggle API read, genuine mini/full datasets, no secret in Git, documented handoffs | `evaluation/00_preparation.md` |
| **Phase 1 — complete** | Audit inputs and freeze evaluation contract | Verified annotations/images, checkpoint provenance, ID mappings and explicit protocol | `evaluation/01_protocol_and_inputs.md` |
| **Phase 2 — complete** | Export reliable COCO predictions | Schema/mapping checks, exact evaluated image set, reproducible output | `evaluation/02_prediction_export.md` |
| **Phase 3 — complete** | Implement and validate micro-scale evaluator | Meaningful analytic fixtures, AP/AR/PR and operating-point precision/recall | `evaluation/03_evaluator_validation.md` |
| **Phase 4 — harness complete; benchmarks pending** | Integrate CLI and run two baselines | CLI checks, consistent runs, machine-readable results and provenance | `evaluation/04_cli_and_baselines.md` |
| **Phase 5 - tooling/handoff complete; trained conclusions pending** | Explain degradation with figures and examples; prepare handoff | Tables, class-size diagnostics, error gallery and reproducible final benchmark | `evaluation/05_analysis_and_handoff.md` |

### Phase 1 — Audit inputs and freeze the evaluation protocol

- [x] Inventory real dataset files and record the pending **AI-TOD-trained** checkpoint handoff. Capture checkpoint provenance when artifacts arrive; its current absence does not block harness implementation. Do not substitute a COCO-pretrained checkpoint as an accuracy baseline.
- [x] Hash annotation JSONs and checkpoints; record repository revision, Python/library versions, device and run settings. Check annotation split/version, actual image count, unique IDs, category names/IDs, invalid geometry, area-vs-box-area consistency, ignore/crowd flags and train/validation disjointness.
- [x] Match every selected original image filename to its COCO ID; check duplicate stems, nested paths, dimensions and actual readable imagery. Never infer IDs from numeric filenames or directory order.
- [x] Audit the prepared **1,000 train / 200 val**, seed 42 development splits and `data/ai-tod-v2/preparation_summary.json`; regenerate only if the input audit requires it. Quantify distribution drift; stratification is approximate and rare-class boosting can shift it. The preparation already uses genuine imagery without placeholders and preserves selected IDs, annotation hashes and sampling settings.
- [x] Verify full validation coverage independently. A mini result is a development result; final comparisons use the same agreed full validation set. Do not reconstruct official evaluation ground truth from rounded YOLO labels.
- [x] Freeze assignment vs upstream area settings, IoUs **0.50:0.05:0.95**, 101 recall points, category-aware matching, max detections, export confidence floor and postprocessing. Frozen export confidence floor: **0.001**, subject to prediction-based truncation verification in Phase 2. Avoid the project's generic inference confidence 0.25 for AP curves.
- [x] Define an operating-point score threshold (**0.25**, fixed for both baselines), IoU 0.50 precision/recall, and separately the average of per-threshold values across 0.50:0.05:0.95. Describe score selection without tuning on the final evaluation outcomes. Keep operating-point recall distinct from COCO AR.
- [x] Write the protocol and input audit, including unresolved decisions. Stop for your review before Phase 2.

**Gate:** Inputs and protocol are explicit; fixture/pretrained smoke tests are labeled separately from trained accuracy results. Real benchmark results must trace to genuine images, known annotations, exact image IDs and a named trained checkpoint. The pending trained weights do not block this phase or schema/evaluator work.

### Phase 2 — Export and validate COCO predictions

- [x] Implement the smallest reusable export adapter under existing `src/eval/`. Check installed Ultralytics behavior before relying on `model.val(save_json=True)`.
- [x] Validate checkpoint class vocabulary against the dataset's eight canonical classes. Map **model class index → class name → original COCO category ID**; reject unknown/ambiguous mappings.
- [x] Export a JSON array of `{image_id, category_id, bbox: [x,y,w,h], score}`. Boxes must be finite positive dimensions in **original-image pixel coordinates**, with letterbox/resize offsets undone.
- [x] Preserve evaluated image IDs separately, including images with **zero predictions**; evaluate the whole selected set, not only IDs appearing in detection JSON.
- [x] Record confidence floor, image size, batch/device, NMS or end-to-end behavior, detection limits and timing. Keep confidence/postprocessing consistent where possible; document model-specific behavior rather than silently forcing it to match.
- [x] Check a few overlaid detections against original images, plus ID/category mapping fixtures. Handle an empty result list, invalid results, and duplicate file stems explicitly.
- [x] Save reusable predictions and metadata under `runs/eval/<run_name>/`; write Phase 2 report and pause.

**Gate:** No fabricated IDs, silent class mismatch, image omission or coordinate-system mismatch.

### Phase 3 — Implement and verify the evaluator

- [x] Added **pycocotools 2.0.11** as an explicit dependency with updated `uv.lock`. The evaluator wraps COCOeval and reuses its matching semantics.
- [x] Set area ranges **and labels** explicitly. Extract AP/AR from accumulated tensors by label and IoU; do not trust standard `summarize()` positions after changing the bins.
- [x] Produce overall AP@[.50:.95], AP50, AP75, APvt/APt/APs/APm, bin-wise recall/AR, per-class × size AP/support, and PR data. Record AP display units (fraction vs percentage) consistently.
- [x] Exclude `-1` tensor entries from means; report **N/A** for bins/classes without eligible ground truth. Distinguish a supported class with no true positives (**0**) from missing support (**N/A**).
- [x] Compute operating-point TP/FP/FN from score-thresholded evaluator matches and ignore flags at each requested IoU; define macro/micro aggregation and zero-denominator behavior. Do not read COCO's interpolated precision as fixed-confidence precision. Respect crowd/ignore matching and size handling for unmatched detections.
- [x] Add small meaningful tests: perfect predictions; no detections; wrong class; duplicates; localized shifts; sparse/non-contiguous IDs; empty-support categories; crowd/ignored GT; exact bin boundaries; low-score detections; detection caps; images without predictions.
- [x] Verify all-area agreement with standard COCOeval for identical settings and micro-bin agreement against a pinned upstream reference when using that reference's settings. Do not claim parity from matching only the overall AP.
- [x] Write Phase 3 report, including any correction of protocol assumptions; pause.

**Gate:** Analytically predictable cases pass and unsupported slices remain visible instead of becoming artificial zeros.

### Phase 4 — CLI integration and fair baseline runs

- [x] Register `@eval_app.command("run")` in existing `src/cli.py`; required interface: `--weights`, `--data`, `--split`. Also expose annotation JSON, optional existing predictions, output directory, device, protocol/detection cap and operating-point threshold as needed.
- [x] Keep the evaluator usable from COCO JSON without running a model, so inference/architecture/loss collaborators can reuse it. Print results with `src.utils.ui.print_table` and existing header/status helpers.
- [x] Verify CLI help, successful fixture-based evaluation, clear invalid-input errors and deterministic repeated evaluation. Check required arguments and unsupported split values.
- [x] Run real-image smoke CLI evaluations in saved-JSON and checkpoint modes; verify both models and exact image coverage.
- [ ] Evaluate both **genuine trained** baselines on the same **2,804 full validation IDs** at **800 px**. Pending: checkpoints/training provenance; the shared Kaggle status query was denied.
- [x] Save predictions, scalar metrics, PR arrays, per-class/size support, metadata and command settings. Reuse saved predictions for plotting/evaluator debugging rather than rerunning inference unnecessarily.
- [x] Verify requested caps and recorded cap hits, class vocabulary, augmentation settings and genuine imagery on executed smoke paths. Timing remains context, not a standardized latency benchmark.
- [ ] Inspect cap/confidence sensitivity for the genuine trained full-validation outputs before final benchmark completion.
- [x] Write Phase 4 report and pause.

**Gate:** Two comparable and reproducible baseline evaluations, or explicitly identified missing checkpoint/data prerequisites; no invented benchmark entries.

**Checkpoint handoff handling:** If final checkpoints are still pending, report **CLI/harness complete; final benchmark pending**, rather than marking baseline evaluation complete. Member 1 may execute the official runs and supply prediction/metric artifacts for Member 2's analysis. At the phase review, decide whether to validate plotting code with labeled fixtures while waiting; final degradation conclusions still require real baseline results.

### Phase 5 — Analysis, visual evidence and handoff

- [x] Created `docs/evaluation_baselines.md` with an explicitly labeled measured smoke snapshot, provenance, complete AP/AR table, and final benchmark gate. Genuine trained values remain unfilled.
- [x] Quantify **absolute AP gaps** in percentage points and **relative losses** versus a stated supported reference bin, e.g. $100(AP_m-AP_{vt})$ points and $100(1-AP_{vt}/AP_m)$ percent. Mark relative loss undefined when the reference is zero/N/A. Do not assume monotonic degradation.
- [x] Separate object-size trends from class composition using class × size results. Show support counts so a rare or empty bin does not drive a broad conclusion. Do not average bin APs as a substitute for overall COCO AP.
- [x] Generate paired model-by-bin AP plots, size-specific PR curves (clearly label IoU 0.50 or mean over IoUs), recall plots with threshold labels, and class × size heatmaps with support/N/A.
- [x] Select representative missed and localization failure examples using recorded matching criteria; show original image context and enlarged crops, GT/predicted boxes, class/score, scale and IoU. Label the sample as illustrative, not a population error rate. Leave full TIDE attribution to the failure-analysis owner.
- [x] Explain practical implications for aerial vehicle/person monitoring: missed small objects, tolerance to localization shifts and operating-point trade-offs. Avoid deployment/safety claims unsupported by this dataset.
- [x] Provide reusable prediction/metric schemas to inference, architecture/loss and report owners; document how they evaluate new ablations with this same harness.
- [x] Check relevant tests, CLI examples and links, remove temporary helpers, review Git diff and ensure credentials/data/weights are excluded. Prepare PR content; follow your explicit phase-end instructions before any publication.
- [x] Wrote Phase 5 report and pause for final implementation/handoff review; final trained conclusions remain pending.

**Implementation gate: passed on explicitly labeled smoke evidence.** Comparison/gap tooling, support-aware figures, match-based examples, schemas and rerun commands are complete.

- [ ] **Final scientific gate:** replace the smoke snapshot with two genuine trained full-validation runs, review training provenance and cap sensitivity, then confirm the final size-degradation conclusions. This remains pending external artifacts.

The current implementation uses one class-controlled gap report at a time (`--category-id`, `--reference-bin`); it does not certify training provenance or perform formal TIDE attribution. See [Phase 5 report](evaluation/05_analysis_and_handoff.md).

## 4. Figure and reporting conventions

Reuse `src/utils/plot_style.py`: **`set_publication_style`, `add_header`, `clean_spines`, `save_figure`, `COLORS`, `SCALE_BIN_PALETTE`**. Keep existing Arial/Helvetica/DejaVu Sans fallback, sans-serif Mathtext, white background, horizontal dashed grids, clean spines and **300 dpi** output. Use scale-bin colors consistently; distinguish models by position or line style. Reserve header space, label units/protocol, and inspect actual saved images for clipping and legibility.

Aim for **2–3 complementary figures per phase** when evidence is available. Each should answer a distinct question and include **quantified pattern/comparison, interpretation, practical implication and a support/confounding check**. Figures are created when actual phase evidence warrants them. Store durable report figures under **`assets/eval/`** (create only when first needed), link them relatively from reports, and keep raw arrays/predictions in ignored **`runs/eval/`**. Do not re-run `sod data eda` with its default report/figure paths during onboarding, because that overwrites the team's existing EDA deliverables.

After each phase, use the report structure in [evaluation/README.md](evaluation/README.md): **objective → method → code/check results → numerical analysis → visualization analysis → practical implications → limitations → next decision**. Use bold keywords, comparison tables and mathematical conclusions with $\implies$. If a visualization or numerical result is not applicable, say why; never create placeholder measurements. Update checkboxes only after evidence exists.

## 5. Minimal file policy

| Location | Planned use | Creation policy |
| :--- | :--- | :--- |
| `docs/member2_working_plan.md` | This tracker | Created during preparation |
| `docs/evaluation/` | Report index and one completed report per phase | Created now; future reports only after phase work |
| Existing `src/eval/` | Phase 1 protocol/audit modules; export and evaluator follow | Add only during the relevant implementation phase |
| Existing `src/cli.py` | Evaluation command | Extend in Phase 4 |
| `pyproject.toml`, `uv.lock` | Explicit evaluator dependency | Update in Phase 3 |
| `tests/` | Contract tests added in Phase 1; evaluator matching tests follow in Phase 3 | Create tests only for meaningful correctness cases |
| `assets/eval/` | Report figures | Create when measured figures exist |
| Ignored `data/`, `runs/` | Datasets, predictions, checkpoints and metadata | Keep large/private artifacts out of Git |

Avoid parallel report systems, standalone checking scripts and redundant configuration files. Reuse existing modules, and delete temporary inspection/render/bootstrap artifacts after their purpose is complete.
