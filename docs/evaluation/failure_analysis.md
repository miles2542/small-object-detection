# Detection failure analysis with TIDE

`sod eval failures` analyzes saved COCO prediction JSON without running a model. It uses
the original [`tidecv` 1.0.1](https://github.com/dbolya/tide) error attribution engine
and the repository's AI-TOD COCO evaluator for official size-bin AP.

## Inputs

- A COCO ground-truth JSON with `images`, `categories`, and `annotations`.
- One or more prediction JSON arrays in the format documented in
  [the evaluation guide](README.md#prediction-json-schema).
- Optional validation images under `--image-root` for a qualitative gallery.

```bash
sod eval failures \
  --ground-truth runs/eval/input_audit/val_ground_truth.json \
  --model baseline=runs/eval/yolov8s/prediction_export/predictions.json \
  --model improved=runs/infer/sahi/predictions.json \
  --image-root data/ai-tod-v2/images/val \
  --output-dir runs/eval/failure_comparison
```

To analyze one model, pass one `--model`. Omit `--image-root` when images are unavailable.
Input JSON paths are never rewritten. Use a new output directory for each analysis.

## Outputs

| File | Meaning |
| :--- | :--- |
| `failure_analysis.json` | Machine-readable input hashes, AP, TIDE oracle ΔAP, event counts, size breakdown, warnings |
| `report.md` | Short numerical report for a paper or presentation draft |
| `tide_breakdown.png` | Six TIDE error types by oracle ΔAP50 and count |
| `failure_by_size.png` | Percentage composition of error events within each exclusive size bin |
| `failure_gallery.png` | Full-frame and enlarged examples for Miss, Loc, Bkg, when images are supplied |
| `gallery_examples.json` | IDs, boxes, scores, and source indices behind gallery selection |

## Interpretation

TIDE uses foreground IoU 0.50 and background IoU 0.10. Its six modes are `Cls`
(classification), `Loc` (localization), `Both`, `Dupe`, `Bkg` (background), and `Miss`.
The oracle ΔAP50 values are independent counterfactual improvements in percentage
points; **do not sum them**. TIDE's AP50 is shown beside the official COCO AP over
IoU 0.50:0.95 by AI-TOD size bin. These are different metrics; TIDE's matching and
handling of categories without ground-truth support can also make its AP50 differ
from the repository's COCO AP50.

For an interpretable percentage chart, each error event belongs to exactly one size
group: `[0,64)`, `[64,256)`, `[256,1024)`, or `[1024,∞)` square pixels. These
half-open diagnostic bins differ at exact boundaries from the official closed AP
bins, where an object at 64, 256, or 1024 px² belongs to both neighboring bins.
`Cls`, `Loc`, `Dupe`, and `Miss` are assigned by the linked ground-truth area;
`Both` and `Bkg` use predicted box area. TIDE `Miss` excludes a ground-truth miss
already explained by another error mode, so it is not the same as COCO false negatives.
The background chart has predicted-box size, not a hidden ground-truth size.

Localization shift is the distance between box centers for `Loc` errors, reported
in original image pixels. The gallery picks the first eligible image and annotation
ID deterministically in each size/error group. It is illustrative, not a random
sample or a causal explanation.

When a sibling `manifest.json` exists, its prediction and annotation SHA-256 hashes,
image IDs, and counts are checked. Models with verified manifests must cover the
same image IDs. Missing manifests or differing export settings are recorded as
comparability warnings. Keep the confidence export floor low enough to retain
candidate detections; TIDE analyzes the saved ranked predictions and cannot recover
boxes omitted before export. A smoke-only or partial validation split should not
be reported as a full benchmark.
