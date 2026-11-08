"""Pure NumPy vectorized Weighted Boxes Fusion (WBF) implementation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _vectorized_iou(box: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Compute IoU between a candidate box [4] and multiple cluster boxes [K, 4]."""
    x1 = np.maximum(box[0], clusters[:, 0])
    y1 = np.maximum(box[1], clusters[:, 1])
    x2 = np.minimum(box[2], clusters[:, 2])
    y2 = np.minimum(box[3], clusters[:, 3])

    intersection_w = np.maximum(0.0, x2 - x1)
    intersection_h = np.maximum(0.0, y2 - y1)
    intersection = intersection_w * intersection_h

    box_area = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    clusters_area = np.maximum(0.0, clusters[:, 2] - clusters[:, 0]) * np.maximum(
        0.0, clusters[:, 3] - clusters[:, 1]
    )

    union = box_area + clusters_area - intersection
    return np.where(union > 0.0, intersection / union, 0.0)


def weighted_boxes_fusion(
    boxes_list: Sequence[np.ndarray | Sequence[Sequence[float]]],
    scores_list: Sequence[np.ndarray | Sequence[float]],
    labels_list: Sequence[np.ndarray | Sequence[int]],
    weights: Sequence[float] | None = None,
    iou_threshold: float = 0.50,
    skip_box_threshold: float = 0.05,
    conf_type: str = "avg",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse multi-model bounding box predictions via Weighted Boxes Fusion.

    Parameters
    ----------
    boxes_list : Sequence[np.ndarray]
        List of M bounding box arrays, each shaped (N_m, 4) in [x1, y1, x2, y2] order.
    scores_list : Sequence[np.ndarray]
        List of M confidence score arrays, each shaped (N_m,).
    labels_list : Sequence[np.ndarray]
        List of M category label arrays, each shaped (N_m,).
    weights : Sequence[float] | None
        Confidence weighting factor per model. Defaults to uniform weights [1.0] * M.
    iou_threshold : float
        Intersection-over-Union threshold for clustering overlapping detections.
    skip_box_threshold : float
        Floor confidence below which individual detections are excluded prior to fusion.
    conf_type : str
        Consensus score computation strategy:
        - "avg": Arithmetic mean of member confidence scores within the cluster.
        - "model_scaled": Cluster confidence sum normalized by total model count M.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        fused_boxes (N, 4), fused_scores (N,), fused_labels (N,) sorted descending by score.
    """
    valid_conf_types = {"avg", "model_scaled"}
    if conf_type not in valid_conf_types:
        raise ValueError(
            f"Unsupported conf_type: '{conf_type}'. Must be one of {sorted(valid_conf_types)}"
        )

    num_models = len(boxes_list)
    if num_models != len(scores_list) or num_models != len(labels_list):
        raise ValueError(
            f"Length mismatch: boxes_list ({num_models}), scores_list ({len(scores_list)}), "
            f"labels_list ({len(labels_list)}) must all share identical length"
        )

    empty_return = (
        np.empty((0, 4), dtype=float),
        np.empty((0,), dtype=float),
        np.empty((0,), dtype=int),
    )
    if num_models == 0:
        return empty_return

    if weights is None:
        norm_weights = np.ones(num_models, dtype=float)
    else:
        if len(weights) != num_models:
            raise ValueError(
                f"Weights length ({len(weights)}) does not match model count ({num_models})"
            )
        w_arr = np.asarray(weights, dtype=float)
        if np.any(w_arr < 0.0):
            raise ValueError("Model weights must be non-negative")
        sum_w = float(np.sum(w_arr))
        if sum_w <= 0.0:
            raise ValueError("Sum of model weights must be strictly positive")
        norm_weights = w_arr / sum_w * float(num_models)

    # Validate and filter predictions per model
    filtered_boxes: list[np.ndarray] = []
    filtered_scores: list[np.ndarray] = []
    filtered_labels: list[np.ndarray] = []

    for m in range(num_models):
        b = np.asarray(boxes_list[m], dtype=float)
        s = np.asarray(scores_list[m], dtype=float)
        l = np.asarray(labels_list[m], dtype=int)

        if b.size == 0:
            b = np.empty((0, 4), dtype=float)
        if s.size == 0:
            s = np.empty((0,), dtype=float)
        if l.size == 0:
            l = np.empty((0,), dtype=int)

        if b.ndim != 2 or (b.shape[0] > 0 and b.shape[1] != 4):
            raise ValueError(f"Model {m} boxes must have shape (N, 4), got {b.shape}")
        if len(b) != len(s) or len(b) != len(l):
            raise ValueError(
                f"Model {m} length mismatch: {len(b)} boxes, {len(s)} scores, {len(l)} labels"
            )

        if len(b) > 0:
            mask = s >= skip_box_threshold
            b = b[mask]
            s = s[mask] * norm_weights[m]
            l = l[mask]

        filtered_boxes.append(b)
        filtered_scores.append(s)
        filtered_labels.append(l)

    total_valid = sum(len(b) for b in filtered_boxes)
    if total_valid == 0:
        return empty_return

    unique_labels = np.unique(
        np.concatenate([l for l in filtered_labels if len(l) > 0])
    )

    all_fused_boxes: list[np.ndarray] = []
    all_fused_scores: list[float] = []
    all_fused_labels: list[int] = []

    for label in unique_labels:
        label_boxes_list: list[np.ndarray] = []
        label_scores_list: list[np.ndarray] = []

        for m in range(num_models):
            if len(filtered_labels[m]) == 0:
                continue
            idx = np.where(filtered_labels[m] == label)[0]
            if len(idx) > 0:
                label_boxes_list.append(filtered_boxes[m][idx])
                label_scores_list.append(filtered_scores[m][idx])

        if not label_boxes_list:
            continue

        cat_boxes = np.concatenate(label_boxes_list, axis=0)
        cat_scores = np.concatenate(label_scores_list, axis=0)

        # Sort descending by weighted score
        sort_order = np.argsort(-cat_scores)
        cat_boxes = cat_boxes[sort_order]
        cat_scores = cat_scores[sort_order]

        cluster_centers: list[np.ndarray] = []
        cluster_boxes: list[list[np.ndarray]] = []
        cluster_scores: list[list[float]] = []

        for box, score in zip(cat_boxes, cat_scores, strict=True):
            if not cluster_centers:
                cluster_centers.append(box.copy())
                cluster_boxes.append([box])
                cluster_scores.append([score])
                continue

            centers_array = np.array(cluster_centers, dtype=float)
            ious = _vectorized_iou(box, centers_array)
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])

            if best_iou >= iou_threshold:
                cluster_boxes[best_idx].append(box)
                cluster_scores[best_idx].append(score)

                m_b = np.array(cluster_boxes[best_idx], dtype=float)
                m_s = np.array(cluster_scores[best_idx], dtype=float)
                s_sum = float(np.sum(m_s))
                if s_sum > 0.0:
                    cluster_centers[best_idx] = np.sum(m_b * m_s[:, None], axis=0) / s_sum
                else:
                    cluster_centers[best_idx] = np.mean(m_b, axis=0)
            else:
                cluster_centers.append(box.copy())
                cluster_boxes.append([box])
                cluster_scores.append([score])

        for c_boxes, c_scores in zip(cluster_boxes, cluster_scores, strict=True):
            m_b = np.array(c_boxes, dtype=float)
            m_s = np.array(c_scores, dtype=float)
            s_sum = float(np.sum(m_s))

            if s_sum > 0.0:
                fused_box = np.sum(m_b * m_s[:, None], axis=0) / s_sum
            else:
                fused_box = np.mean(m_b, axis=0)

            if conf_type == "avg":
                fused_score = s_sum / float(len(c_scores))
            else:
                fused_score = s_sum / float(num_models)

            fused_score = min(max(float(fused_score), 0.0), 1.0)
            all_fused_boxes.append(fused_box)
            all_fused_scores.append(fused_score)
            all_fused_labels.append(int(label))

    if not all_fused_boxes:
        return empty_return

    final_boxes = np.array(all_fused_boxes, dtype=float)
    final_scores = np.array(all_fused_scores, dtype=float)
    final_labels = np.array(all_fused_labels, dtype=int)

    final_order = np.argsort(-final_scores)
    return final_boxes[final_order], final_scores[final_order], final_labels[final_order]
