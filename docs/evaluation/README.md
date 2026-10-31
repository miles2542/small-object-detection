# Evaluation integration and team handoff

**Integration branch:** [`feat/m2-evaluation-coco`](https://github.com/miles2542/small-object-detection/tree/feat/m2-evaluation-coco).

**Start here when evaluating a baseline, P2 model, SAHI output or NWD ablation.** This folder documents the shared measurement contract, verified implementation and evidence limits. Read the [benchmark status](../evaluation_baselines.md) before quoting any score.

## 1. What is ready, and what remains open

| Surface | Current state | What a teammate can do now |
| :--- | :--- | :--- |
| Dataset/input audit and original-ID adapter | **Implemented and verified** | Reuse exact GT and selected image IDs |
| Native `.pt` prediction export | **Implemented and verified** | Export trained checkpoints; preserve complete image coverage and metadata |
| Micro-scale COCO evaluator | **Implemented and verified** | Evaluate original-ID JSON with assignment or explicit upstream ranges |
| `sod eval run` | **Implemented and verified** | Run checkpoint inference or reuse saved JSON |
| Comparison/gap/failure-example analysis | **Implemented and verified** | Compare two compatible completed runs without repeating inference |
| Published numerical examples | **Pretrained smoke diagnostics only** | Understand interfaces and trends; do not quote them as trained baselines |
| Genuine trained full-validation benchmark | **Pending checkpoint/training-provenance handoff** | Run the documented commands once real artifacts exist |

**25 test methods pass**; repository-source Ruff checks pass. The evaluator's complete precision/recall/score tensors matched a pinned AI-TOD reference on the tested fixtures and saved smoke inputs; details and limits are in Phase 3. Successful integration does not certify a new checkpoint's training quality or an untested environment.

### Role note, especially for Member 3

The repository [team quickstart](../team_quickstart.md) assigns **Member 3 to P2 architecture** and **Member 4 to SAHI**; the assignment materials associate Member 3 with inference and differ in later numbering. Use the team's agreed responsibility rather than relying only on member numbers. This guide supplies both handoffs:

- **P2/architecture owner, including Member 3 under the quickstart:** use the checkpoint path in Section 3, keeping the original evaluation data/protocol and eight class names.
- **Inference/SAHI owner, including Member 3 under the assignment:** use the original-coordinate COCO contract and saved-JSON path in Section 4; document tile merging and preserve empty-image coverage.

Phase reports preserve their original completion checkpoints; the status table above and benchmark status page describe the current state. Later figure/analysis revisions are identified inside the reports.

## 2. Read only what your next task needs

| Report | Verified contribution | Must-know finding / integration use |
| :--- | :--- | :--- |
| [00 - Preparation](00_preparation.md) | Environment and real data layout | Data, weights and run outputs are intentionally ignored by Git |
| [01 - Protocol and inputs](01_protocol_and_inputs.md) | Source hashes, GT sanitation, subset drift; **three figures** | Full validation is 2,804 images / **75,082 eligible GT**; closed bins overlap; dense-scene caps matter |
| [02 - Prediction export](02_prediction_export.md) | Class/coordinate/coverage adapter; **three figures** | Original IDs; zero outputs; active native-head cap; score-distribution and mode sensitivity |
| [03 - Evaluator validation](03_evaluator_validation.md) | Matching, AP/AR/PR, operating counts, reference parity | Interpolated PR is not fixed-confidence precision; unsupported slices are N/A |
| [04 - CLI integration](04_cli_and_baselines.md) | Executed checkpoint/JSON modes, numerical equivalence | Correct YAML selection versus explicit JSON selection; smoke vocabulary is incomplete |
| [05 - Analysis and handoff](05_analysis_and_handoff.md) | Signed gaps, support-aware comparisons, deterministic error examples | Reference/support matter; illustrative failures are not causal/TIDE attribution |

Also see the [working plan](../member2_working_plan.md), [portable numerical/status handoff](../evaluation_baselines.md), and [team environment/data guide](../team_quickstart.md).

## 3. Checkpoint route: P2, NWD and trained baselines

After checkout, synchronize the environment with **`uv sync --locked`**, activate `.venv`, and prepare real imagery using the team quickstart. For full-size preparation, `--train-size 0 --val-size 0` selects full splits. Use **`--no-placeholders`** to fail on missing imagery rather than create synthetic files. Confirm the actual source images exist before evaluating.

```powershell
# Inspect the command; Windows can also use .\sod.cmd without activation.
sod eval run --help

# Development integration check with a genuine eight-class trained checkpoint.
# Replace PATH_TO_P2_BEST.pt with an actual existing local file.
sod eval run --weights PATH_TO_P2_BEST.pt --data configs/aitod_mini.yaml --split val --output-dir runs/eval/p2_mini

# Final comparable validation, once the trained artifacts are available.
sod eval run --weights PATH_TO_BASELINE_BEST.pt --data configs/aitod_full.yaml --split val --output-dir runs/eval/baseline_full
sod eval run --weights PATH_TO_P2_BEST.pt --data configs/aitod_full.yaml --split val --output-dir runs/eval/p2_full

# Reuse both completed runs; no new model inference.
python -m src.eval.analysis --run Baseline=runs/eval/baseline_full --run P2=runs/eval/p2_full --output-dir runs/eval/p2_comparison --image-root data/ai-tod-v2/splits/full/images/val
```

Choose **fresh output directories**. Add `--device 0` for an available compatible CUDA environment; use `--device cpu --batch-size 4` for local smoke/development work. Device changes need documenting; this guide's CPU runs do not establish standardized latency comparisons.

**P2/NWD owner:** provide the actual `.pt`, SHA-256, exact class names/order and training config/logs: train selection, epochs, seed and changed architecture/loss. The exporter maps model indices by **exact class names**, so an index reorder is acceptable only with correct unique names. Missing/unknown names fail in ordinary evaluation. A new head still needs active-backend/native-cap verification; a modified model must produce compatible detection `Results` rather than silently changing output semantics. Feature-cell size is motivation for P2, not proof that a layer has zero activation or that AP must improve.

YAML mode supports a **single image directory per split**, not lists or `.txt` split files. It uses actual filenames to select source COCO IDs; unknown files, ambiguous roots and class-name disagreements fail. Standard prepared train/val annotations are inferred. Other layouts use `--annotations COCO_JSON`; explicit COCO input with weights additionally needs `--image-root`. JSON image records define the image set: a `--split val` label cannot convert a training JSON to validation data.

## 4. Inference/SAHI route: exact COCO and coverage contract

For custom inference, supply a COCO results array:

```json
[{"image_id": 91, "category_id": 17, "bbox": [64.0, 32.0, 128.0, 64.0], "score": 0.75}]
```

This is an **analytic schema example**, not an AI-TOD measurement. Actual AI-TOD IDs must come from its own annotation file.

| Field / condition | Required meaning | Typical integration error |
| :--- | :--- | :--- |
| `image_id` | Original integer COCO image ID from the exact split | Numeric filename stem, tile ID, enumeration index or train/val ID mix-up |
| `category_id` | Original integer COCO category ID via exact class-name mapping | Model class index copied directly, guessed boat/ship or car/vehicle aliases |
| `bbox` | `[x,y,width,height]`, original-image pixel coordinates, positive dimensions | XYXY/normalized values, tile-local coordinates, repeated resize inversion |
| `score` | Finite probability-like confidence in [0,1], preserve ranking precision | Rounded scores, NaN, or a high display cutoff applied before AP export |
| Image coverage | Exact GT image list, including negative/zero-prediction images | Building the evaluated set from nonempty predictions |
| All-empty run | A valid JSON array `[]`, with the complete GT image set | Missing/truncated file confused with valid emptiness |

**SAHI sequence:** predict tiles → reverse tile resize/padding → add tile offsets into original image coordinates → merge duplicates across overlap → export original image/category IDs. If the installed library already restores original coordinates, do not apply the transform again. State tile dimensions, overlap, per-tile floor/cap, merging algorithm/threshold, and final full-image floor/cap. A high per-tile cap cannot repair an earlier low top-k cutoff, and a high final cap cannot restore discarded tile outputs.

```powershell
# Evaluate external merged JSON against the exact selected normalized COCO GT.
sod eval run --data runs/eval/input_audit/val_ground_truth.json --predictions runs/infer/sahi/predictions.json --split val --output-dir runs/eval/sahi_full
```

The example GT is created by **`python -m src.eval.audit`** after full data preparation. That command performs a real image/hash audit and writes normalized full/mini GT; it is not a fast plotting-only command. Pass your actual equivalent exact normalized GT if using a different workspace layout.

**Manifest behavior:** exporter-produced results have a sibling `manifest.json`; it must be complete and its hashes/counts/image coverage must agree. Preserve it alongside the exact annotation JSON. A partial or mismatched manifest is rejected. Plain external JSON is accepted without a manifest but labeled **`external_predictions_unverified`**; this is not a completed provenance handoff. Do not fabricate a "complete" manifest to suppress that label. To reuse native exporter manifests, call the existing `src.eval.export.export_predictions`; for a custom sliced pipeline, document equivalent source IDs, all-image coverage, settings, counts, weights/training provenance and SHA-256 evidence for review.

The analyzer compares **two completed CLI run directories**, not raw prediction files. It rejects different annotation hashes, image/category IDs or evaluation protocols; available export settings/vocabularies must also agree. An inference experiment that intentionally changes resolution or postprocessing needs its own clearly documented controlled comparison. The current automatic analyzer's strict setting guard must not be bypassed by editing manifests. External/unverified inputs remain visibly unverified, and unknown export floors do not receive an assumed low-score sweep.

## 5. Frozen measurement contract and must-know cautions

| Topic | Contract / caution |
| :--- | :--- |
| Default protocol | **`assignment`**: area all `[0,1e10]`; vt `[4,64]`, t `[64,256]`, s `[256,1024]`, m `[1024,4096]` px² |
| Alternative | **`upstream`** changes vt lower bound to 0 and medium upper bound to 1e10; label separately, never mix the two profiles |
| Scale | $\sqrt{w h}$ in original pixels; not maximum side, feature-map cells or resized input coordinates |
| Boundaries | Closed on both ends; shared endpoint GT belongs to both adjacent micro groups. Do not sum their support or average their AP to get overall AP |
| GT sanitation | Copy only; remove nonpositive geometry/area, assign positive annotation IDs and preserve source IDs. Keep original positive coordinates/area, including out-of-frame GT |
| Annotation ID zero | Reassigned because COCO matching uses zero as the unmatched sentinel. **Image/category ID zero remains valid** |
| Confidence | Export floor **.001**; fixed operating summary **.25**. `--confidence` changes operating summaries, not the AP ranking. Below a recorded export floor is rejected |
| Caps | Export **1,500 globally/image**; evaluator `[1,100,1500]` **per image/category**. Check active native top-k and cap hits, not only a passed CLI option |
| Model backend | V8 class-aware NMS IoU .70; YOLO26 native end-to-end selected/verified explicitly. Mode diagnostic outputs are not intended native baselines |
| AP/AR | AP averages eligible categories/IoUs/recall samples; AR is maximum-recall based. Neither is pooled thresholded operating recall |
| PR and operating points | COCO PR is interpolated; fixed-score TP/FP/FN come from actual evaluator matches/ignore flags |
| Missing support | JSON `null` / console N/A; supported zero AP stays zero. Empty predictions do not make GT support disappear |
| Macro precision | Supported class with no predictions is defined as zero for macro aggregation; unsupported classes excluded. A pooled zero denominator remains N/A |
| Smoke mode | Explicit `--smoke` allows shared-class pretrained diagnostics. Current outputs have person/airplane only; they are **not eight-class baselines** |
| Evidence limit | No trained final scores, architectural causation, threshold recommendation or deployment suitability is established by the smoke reports |

**Current data caveat:** downloaded v2 validation has **75,091 raw annotations**, nine invalid, hence **75,082 eligible**. The older EDA count of 70,424 describes a different surface. Match source hashes, not filenames alone. Mini validation has **200 images / 5,054 GT** and measurable class drift; full and mini metrics are not interchangeable. Byte-hash disjointness does not prove geographic independence.

## 6. What to send to the next owner

| Artifact | Required contents / interpretation |
| :--- | :--- |
| Model/training evidence | Checkpoint SHA-256, training split/config/epochs/seed/logs and class names; genuine baseline identity |
| Exact GT + selection | Original source identity, selected image IDs including empty outputs; normalized GT actually evaluated |
| Predictions + export evidence | Original-coordinate COCO JSON, complete manifest when produced by the adapter, hashes, mapping, mode and all inference/merge/filter settings |
| CLI run directory | `run.json` plus `evaluation/metrics.json` and `evaluation/curves.npz`; retain full precision |
| Analysis directory | `analysis.json`, generated `comparison.md`, AP/gap PNGs and optional original-image/crop failure figure |
| Interpretation | Split/profile/support/units, explicit reference bin, percentage-point versus relative loss, caps, scope and unresolved limitations |

`metrics.json` stores fractions; console/generated tables display percent. `curves.npz` precision axes are `[IoU, recall, category, area, cap]`, recall axes `[IoU, category, area, cap]`, with named axis arrays. Unsupported tensor entries are `-1`, not measured zero. Relative AP loss is undefined for a zero/missing reference; negative signed gaps indicate improvements. The default analysis reference **person-small (30 mini GT)** is exploratory; medium has only one mini person GT and is not a stable general reference.

The new Phase 2 coverage figure shows **21/42 GT** on v8/native zero-output images; most are smoke-excluded classes. Do not confuse "no output" with "no GT", or interpret every excluded-class miss as person/airplane recognition failure. Likewise, native has fewer boxes at .001 but more at .25 than v8: raw volume or confidence alone cannot rank AP.

Large inputs, pretrained weights, `runs/`, audit hashes and JSON/tensor outputs are **not included in Git**. A fresh clone supplies the code, tests, durable reports and report PNGs. Recreate the environment/data and transfer verified run artifacts separately, rather than expecting local ignored examples to exist after checkout. Do not put API tokens in code, commits or reports.

## 7. Verification before quoting a new result

```powershell
python -m unittest discover -s tests -q
# Repository CI checks source lint with Ruff.
uvx ruff check src
```

Check the reported model/split/profile, exact image set, supported classes, original coordinate space, full denominator, provenance, cap behavior and metric units. Inspect class-by-size support and raw-count operating results before interpreting an aggregate size effect. Verify both new checkpoints on the same held-out full-validation set; inspect training provenance before upgrading status from diagnostic to final benchmark. Final baseline numbers remain explicitly unfilled in the [status page](../evaluation_baselines.md).

For refreshed report figures, reuse **`src/eval/plots.py`** and **`src/utils/plot_style.py`**. Normally provide two or three complementary figures with **quantified pattern → comparison → interpretation → practical implication → support/confounding check**. Use **bold keywords**, comparison tables and $\implies$ conclusions. Keep analytic examples, observed measurements, association and causal attribution distinct. Record failed checks and fixes; remove temporary scripts after use. Preserve the phase-end review convention for further work.
