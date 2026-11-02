# AI-TOD size evaluation: benchmark status and numerical handoff

**Implementation:** Evaluation harness, export adapter, and comparative analysis tooling complete.

**Official benchmark:** **Pending genuine trained-model artifacts.**

**Updated:** 2026-11-01 (Asia/Saigon).

## Official baseline gate

| Required evidence | YOLOv8s | YOLO26s |
| :--- | :--- | :--- |
| Genuine converged eight-class AI-TOD checkpoint + training logs | Pending | Pending |
| Same 2,804 full-validation IDs, source GT and 800-px inputs | Not yet evaluated with genuine checkpoints | Not yet evaluated with genuine checkpoints |
| Official AP/AP50/AP75/APvt/APt/APs/APm and support | **Not measured** | **Not measured** |
| Full-output cap/sensitivity and training provenance review | Pending | Pending |

No score below fills these official entries. COCO-pretrained partial-vocabulary diagnostics retain all eight GT classes but predict only airplane/person. Their purpose is to verify the pipeline and demonstrate analysis; **91.00% of mini-validation annotations belong to the six excluded output classes**.

## Measured smoke snapshot

The following tables were generated from the comparative smoke analysis. Both runs retain **200 images and 5,054 eligible GT**. Eight-class AP averages supported categories, not GT-weighted instances. APvt/APt/APs/APm use the standardized closed micro-scale intervals; displayed values are percentages.

**Status:** smoke_diagnostic_only

AP/AR units: **percent**. These are measured inputs, not proof of training provenance.

| Model | Images | Area | GT | AP | AP50 | AP75 | AR |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8s | 200 | all | 5054 | 0.3675 | 0.9082 | 0.2067 | 1.6471 |
| YOLOv8s | 200 | verytiny | 786 | 0.0417 | 0.2250 | 0.0165 | 0.2500 |
| YOLOv8s | 200 | tiny | 3670 | 0.7699 | 1.9266 | 0.4307 | 2.1380 |
| YOLOv8s | 200 | small | 565 | 0.9185 | 2.2392 | 0.3831 | 5.2381 |
| YOLOv8s | 200 | medium | 89 | 1.1040 | 2.3950 | 0.0000 | 9.4444 |
| YOLO26s | 200 | all | 5054 | 0.6252 | 1.8231 | 0.2787 | 2.5351 |
| YOLO26s | 200 | verytiny | 786 | 0.1554 | 0.7153 | 0.0130 | 0.8750 |
| YOLO26s | 200 | tiny | 3670 | 1.0366 | 3.0364 | 0.4512 | 3.1561 |
| YOLO26s | 200 | small | 565 | 2.0408 | 5.4949 | 1.0850 | 7.1293 |
| YOLO26s | 200 | medium | 89 | 0.3029 | 0.8131 | 0.0000 | 10.0000 |

### Class-controlled gaps versus small

Signed gap = reference AP minus bin AP; negative is improvement. Relative loss is N/A for zero/missing reference.

| Model | Class | Area | GT | AP (%) | Gap (pp) | Relative loss (%) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| YOLOv8s | person | verytiny | 144 | 0.2087 | 6.2206 | 96.7539 |
| YOLOv8s | person | tiny | 221 | 6.1593 | 0.2700 | 4.1996 |
| YOLOv8s | person | small | 30 | 6.4293 | 0.0000 | 0.0000 |
| YOLOv8s | person | medium | 1 | 6.2500 | 0.1793 | 2.7891 |
| YOLO26s | person | verytiny | 144 | 0.7772 | 12.6983 | 94.2328 |
| YOLO26s | person | tiny | 221 | 8.2929 | 5.1825 | 38.4590 |
| YOLO26s | person | small | 30 | 13.4754 | 0.0000 | 0.0000 |
| YOLO26s | person | medium | 1 | 0.8994 | 12.5761 | 93.3260 |

### Input identity

- **YOLOv8s:** `runs/eval/yolov8s_json`; purpose `smoke_only`.
  Metrics SHA-256: `d9cdedf79b1035a38bf4c64fbdd09d799a19b3e33e845cc1ba50af22162e6359`.
  Annotations: `40a8a2f516c7e9ce630340203d80f4c6f036b8b7d6f974349cf85762a4a91d7b`.
  Predictions: `f494a632a6751e93c32053ab1f525a6fdadd9819fd0c404f26074502826d3030`.
  Weights: `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`.
- **YOLO26s:** `runs/eval/yolo26s_json`; purpose `smoke_only`.
  Metrics SHA-256: `852aa33f7b61878f73abea16057b9a9d820a046473b12363955d98552c97c729`.
  Annotations: `40a8a2f516c7e9ce630340203d80f4c6f036b8b7d6f974349cf85762a4a91d7b`.
  Predictions: `02dfd9a98ae342b1e91a3c4ed74dadbd741954ec627b8d3ed12e238edd654675`.
  Weights: `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`.

Inclusive area boundaries overlap; support counts must not be summed as distinct objects.
A small or empty reference can make a ratio unstable or undefined. Inspect support and class composition.

## Reuse and final replacement workflow

1. Obtain the genuine checkpoint files, SHA-256 identities, exact eight class names, training configuration/data/epochs/seed and logs. A file named `best.pt` alone is insufficient provenance.
2. Export/evaluate both through the existing CLI, without `--smoke`:

```powershell
sod eval run --weights PATH_TO_V8_BEST.pt --data configs/aitod_full.yaml --split val --output-dir runs/eval/v8_full
sod eval run --weights PATH_TO_26_BEST.pt --data configs/aitod_full.yaml --split val --output-dir runs/eval/26_full
sod eval compare --run YOLOv8s=runs/eval/v8_full --run YOLO26s=runs/eval/26_full --output-dir runs/eval/full_analysis --image-root data/ai-tod-v2/splits/full/images/val
```

3. Review exact validation coverage, protocol, vocabulary, cap hits, training provenance and supported/empty bins. Inspect lower-cap sensitivity on genuine outputs; the smoke cap observations do not certify future models. The analyzer reproduces numerical inputs but never automatically certifies training provenance.
4. Use the newly generated `comparison.md` and `analysis.json` to update the official table **after review**. Preserve this smoke snapshot as explicitly labeled historical evidence. Regenerate figures on those same inputs; do not reuse smoke captions as final findings.

The same path supports P2/NWD checkpoints. SAHI outputs must merge tile coordinates back into original image coordinates and preserve original image/category IDs, with merging/filtering settings documented. Plain external COCO predictions may be evaluated with their exact GT; without a manifest, provenance remains unverified. Match the resolution, export floor, cap, augmentation and target vocabulary before comparing runs.

## Artifact contract

| Artifact | Required interpretation |
| :--- | :--- |
| Predictions JSON | Array of `{image_id, category_id, bbox:[x,y,w,h], score}`; original pixels and original COCO IDs |
| Export manifest | Selected image IDs including empty outputs; source/weights/prediction hashes; class mapping; settings and cap hits |
| CLI `run.json` | Completion status, exact source/prediction paths and identities, mode, split label, purpose |
| `evaluation/metrics.json` | Fractions, explicit `null` for unsupported metrics; GT support; AP/AR; raw-count operating summaries; provenance |
| `evaluation/curves.npz` | Precision `[IoU,recall,category,area,cap]`, recall `[IoU,category,area,cap]`, scores and axis arrays; unsupported entries `-1` |
| `analysis.json` | Verified comparison, class gaps/reference/support, confidence diagnostics, deterministic example IDs and source hashes |
| `comparison.md` | Portable generated numerical table in percent/percentage points, with scope and input identity |

For a zero/missing AP reference, relative loss is **N/A**. Negative gaps mean improvements. Adjacent closed bins share boundary objects. Class-controlled comparisons help separate composition effects but do not identify causal mechanisms; support and context still matter.

## Documentation & References

- [AI-TOD Micro-Scale Evaluation Guide](evaluation/README.md)
- [Dataset Analysis & Scale Diagnostic Report](dataset_analysis.md)
- [Team Quickstart & Workflow Guide](team_quickstart.md)

**Validation:** 26 test methods pass; Ruff checks pass. The genuine trained benchmark and scientific conclusion gate remains open pending trained checkpoint artifacts.
