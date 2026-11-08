"""Multi-model prediction ensembling engine using Weighted Boxes Fusion."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.postprocess.wbf import weighted_boxes_fusion


def ensemble_predictions(
    prediction_json_paths: Sequence[Path | str],
    weights: Sequence[float] | None = None,
    iou_threshold: float = 0.50,
    conf_threshold: float = 0.25,
    skip_box_threshold: float = 0.001,
    conf_type: str = "avg",
    output_json: Path | str | None = None,
) -> list[dict]:
    """Combine multiple COCO prediction JSON files via Weighted Boxes Fusion.

    Parameters
    ----------
    prediction_json_paths : Sequence[Path | str]
        Paths to COCO prediction JSON files to ensemble.
    weights : Sequence[float] | None
        Per-model confidence weighting factors. Must match length of prediction_json_paths.
    iou_threshold : float
        IoU threshold for clustering boxes during WBF fusion.
    conf_threshold : float
        Minimum score required for a fused prediction to be retained in the final output.
    skip_box_threshold : float
        Individual detection score cutoff before clustering (defaults to 0.001).
    conf_type : str
        Consensus scoring mode: "avg" or "model_scaled".
    output_json : Path | str | None
        Optional destination path to write the fused COCO results JSON.

    Returns
    -------
    list[dict]
        Standard COCO detection records:
        [{"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}, ...]
    """
    if not prediction_json_paths:
        raise ValueError("prediction_json_paths must contain at least one file path")

    paths = [Path(p) for p in prediction_json_paths]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Prediction file not found: {p}")

    num_models = len(paths)
    if weights is not None and len(weights) != num_models:
        raise ValueError(
            f"Weights length ({len(weights)}) does not match prediction files count ({num_models})"
        )

    # Load predictions per model indexed by image_id
    model_image_preds: list[dict[int, list[dict]]] = []
    all_image_ids: set[int] = set()

    for p in paths:
        raw_content = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw_content, list):
            raise TypeError(f"Expected JSON array of detections in {p}, got {type(raw_content)}")

        img_map: dict[int, list[dict]] = defaultdict(list)
        for row in raw_content:
            if not isinstance(row, dict) or not {"image_id", "category_id", "bbox", "score"} <= row.keys():
                raise ValueError(f"Malformed COCO prediction row in {p}")
            img_id = int(row["image_id"])
            img_map[img_id].append(row)
            all_image_ids.add(img_id)

        model_image_preds.append(img_map)

    fused_results: list[dict] = []

    for img_id in sorted(all_image_ids):
        boxes_list: list[np.ndarray] = []
        scores_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []

        for m in range(num_models):
            rows = model_image_preds[m].get(img_id, [])
            if not rows:
                boxes_list.append(np.empty((0, 4), dtype=float))
                scores_list.append(np.empty((0,), dtype=float))
                labels_list.append(np.empty((0,), dtype=int))
                continue

            b_list: list[list[float]] = []
            s_list: list[float] = []
            l_list: list[int] = []

            for r in rows:
                x, y, w, h = r["bbox"]
                b_list.append([float(x), float(y), float(x + w), float(y + h)])
                s_list.append(float(r["score"]))
                l_list.append(int(r["category_id"]))

            boxes_list.append(np.array(b_list, dtype=float))
            scores_list.append(np.array(s_list, dtype=float))
            labels_list.append(np.array(l_list, dtype=int))

        f_boxes, f_scores, f_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_threshold=iou_threshold,
            skip_box_threshold=skip_box_threshold,
            conf_type=conf_type,
        )

        for b, s, l in zip(f_boxes, f_scores, f_labels, strict=True):
            if s < conf_threshold:
                continue

            x1, y1, x2, y2 = b.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0.0 or h <= 0.0:
                continue

            fused_results.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(l),
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "score": round(float(s), 4),
                }
            )

    if output_json is not None:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(fused_results, indent=2), encoding="utf-8")

    return fused_results
