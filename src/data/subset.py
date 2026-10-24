"""Stratified subset sampling for AI-TOD-v2 micro-scale object detection."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

AITOD_CLASSES: list[str] = [
    "airplane",
    "bridge",
    "storage-tank",
    "ship",
    "swimming-pool",
    "vehicle",
    "person",
    "wind-mill",
]
CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(AITOD_CLASSES)}
SCALE_BINS: list[str] = ["vt", "t", "s", "m", "l"]
BIN_TO_INDEX: dict[str, int] = {b: i for i, b in enumerate(SCALE_BINS)}


def get_scale_bin(scale: float) -> str:
    """Map sqrt(w * h) pixel scale to standard AI-TOD micro-scale evaluation bin."""
    if scale <= 8.0:
        return "vt"
    if scale <= 16.0:
        return "t"
    if scale <= 32.0:
        return "s"
    if scale <= 64.0:
        return "m"
    return "l"


def select_stratified_subset(
    coco_data: dict[str, Any],
    target_size: int = 1000,
    seed: int = 42,
    rare_class_boost: bool = True,
) -> tuple[list[int], dict[str, Any]]:
    """Select a representative subset preserving class and scale distributions.

    Preserves:
    - AI-TOD micro-scale bins (AP_vt, AP_t, AP_s, AP_m)
    - Category balance with guaranteed representation for minority classes (wind-mill, bridge, etc.)

    Args:
        coco_data: COCO-format dictionary containing 'images', 'annotations', 'categories'.
        target_size: Desired number of images in subset.
        seed: Random seed for deterministic reproducibility.
        rare_class_boost: Whether to guarantee proportional coverage for rare classes.

    Returns:
        tuple of (selected_image_ids, statistics_dict)
    """
    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])

    num_images = len(images)
    if target_size >= num_images:
        logger.info(
            "Target size %d >= total images %d; selecting all images.",
            target_size,
            num_images,
        )
        all_ids = [img["id"] for img in images]
        return all_ids, {"status": "all_selected", "num_selected": num_images}

    # Map json category IDs to canonical names
    json_cat_map = {c["id"]: c["name"] for c in categories}
    img_id_to_idx = {img["id"]: i for i, img in enumerate(images)}
    idx_to_img_id = {i: img["id"] for i, img in enumerate(images)}

    # Construct feature matrices: (N, 8) categories and (N, 5) scale bins
    img_cat_matrix = np.zeros((num_images, len(AITOD_CLASSES)), dtype=np.int32)
    img_scale_matrix = np.zeros((num_images, len(SCALE_BINS)), dtype=np.int32)
    zero_dim_count = 0

    for ann in annotations:
        bbox = ann.get("bbox", [])
        if len(bbox) < 4:
            zero_dim_count += 1
            continue
        w, h = float(bbox[2]), float(bbox[3])
        if w <= 0.0 or h <= 0.0:
            zero_dim_count += 1
            continue

        raw_cat_id = ann.get("category_id")
        cat_name = json_cat_map.get(raw_cat_id)
        if cat_name not in CLASS_TO_ID:
            continue
        cid = CLASS_TO_ID[cat_name]

        scale = float(np.sqrt(w * h))
        s_bin = get_scale_bin(scale)
        s_idx = BIN_TO_INDEX[s_bin]

        img_id = ann.get("image_id")
        img_idx = img_id_to_idx.get(img_id)
        if img_idx is not None:
            img_cat_matrix[img_idx, cid] += 1
            img_scale_matrix[img_idx, s_idx] += 1

    pop_cats = img_cat_matrix.sum(axis=0).astype(np.float64)
    pop_scales = img_scale_matrix.sum(axis=0).astype(np.float64)
    total_pop_boxes = max(float(pop_cats.sum()), 1.0)

    pop_cat_pct = pop_cats / total_pop_boxes
    pop_scale_pct = pop_scales / total_pop_boxes

    rng = np.random.RandomState(seed)
    selected_mask = np.zeros(num_images, dtype=bool)
    has_boxes = img_cat_matrix.sum(axis=1) > 0

    # Phase 1: Minority class seeding
    if rare_class_boost:
        # Sort classes by population frequency ascending (rarest first)
        rarity_order = np.argsort(pop_cats)
        sampling_ratio = target_size / max(num_images, 1)

        for cid in rarity_order:
            if pop_cats[cid] == 0:
                continue
            # Minimum representation: at least 2 instances or proportional expectation
            expected_quota = max(2, int(np.round(pop_cats[cid] * sampling_ratio)))
            curr_count = img_cat_matrix[selected_mask, cid].sum()

            if curr_count < expected_quota:
                cand_indices = np.where((img_cat_matrix[:, cid] > 0) & (~selected_mask))[0]
                rng.shuffle(cand_indices)
                for idx in cand_indices:
                    if curr_count >= expected_quota:
                        break
                    selected_mask[idx] = True
                    curr_count += img_cat_matrix[idx, cid]

    selected_indices = list(np.where(selected_mask)[0])
    logger.info(
        "Seeded %d images during minority class allocation.",
        len(selected_indices),
    )

    if len(selected_indices) > target_size:
        # Trim while keeping rare classes
        rng.shuffle(selected_indices)
        selected_indices = selected_indices[:target_size]
        selected_mask = np.zeros(num_images, dtype=bool)
        selected_mask[selected_indices] = True

    curr_cats = img_cat_matrix[selected_mask].sum(axis=0).astype(np.float64)
    curr_scales = img_scale_matrix[selected_mask].sum(axis=0).astype(np.float64)

    # Phase 2: Vectorized greedy forward selection to match distributions
    while len(selected_indices) < target_size:
        unselected = np.where((~selected_mask) & has_boxes)[0]
        if len(unselected) == 0:
            # Fall back to any unselected images
            unselected = np.where(~selected_mask)[0]
            if len(unselected) == 0:
                break

        pool_size = min(len(unselected), 300)
        cand_pool = rng.choice(unselected, pool_size, replace=False)

        cand_c = curr_cats + img_cat_matrix[cand_pool]  # (pool_size, 8)
        cand_s = curr_scales + img_scale_matrix[cand_pool]  # (pool_size, 5)
        tot_b = np.maximum(cand_c.sum(axis=1, keepdims=True), 1.0)

        c_dist = cand_c / tot_b
        s_dist = cand_s / tot_b

        # Weighted L1 divergence: balance category alignment and scale distribution
        scores = np.sum(np.abs(c_dist - pop_cat_pct), axis=1) * 2.0 + np.sum(
            np.abs(s_dist - pop_scale_pct), axis=1
        )
        best_cand = cand_pool[np.argmin(scores)]

        selected_mask[best_cand] = True
        selected_indices.append(best_cand)
        curr_cats += img_cat_matrix[best_cand]
        curr_scales += img_scale_matrix[best_cand]

    selected_ids = [idx_to_img_id[idx] for idx in selected_indices]

    # Compute statistical validation report
    sub_cats = img_cat_matrix[selected_mask].sum(axis=0).astype(np.float64)
    sub_scales = img_scale_matrix[selected_mask].sum(axis=0).astype(np.float64)
    tot_sub_boxes = max(float(sub_cats.sum()), 1.0)

    sub_cat_pct = sub_cats / tot_sub_boxes
    sub_scale_pct = sub_scales / tot_sub_boxes

    cat_stats: dict[str, Any] = {}
    for i, cname in enumerate(AITOD_CLASSES):
        cat_stats[cname] = {
            "subset_count": int(sub_cats[i]),
            "subset_pct": round(float(sub_cat_pct[i] * 100), 2),
            "population_count": int(pop_cats[i]),
            "population_pct": round(float(pop_cat_pct[i] * 100), 2),
        }

    scale_stats: dict[str, Any] = {}
    for i, sbin in enumerate(SCALE_BINS):
        scale_stats[sbin] = {
            "subset_count": int(sub_scales[i]),
            "subset_pct": round(float(sub_scale_pct[i] * 100), 2),
            "population_count": int(pop_scales[i]),
            "population_pct": round(float(pop_scale_pct[i] * 100), 2),
        }

    stats: dict[str, Any] = {
        "target_size": target_size,
        "selected_images": len(selected_ids),
        "total_population_images": num_images,
        "selected_boxes": int(tot_sub_boxes),
        "total_population_boxes": int(total_pop_boxes),
        "zero_dim_boxes_filtered": zero_dim_count,
        "category_comparison": cat_stats,
        "scale_bin_comparison": scale_stats,
        "divergence_metric": round(
            float(
                np.sum(np.abs(sub_cat_pct - pop_cat_pct)) * 2.0
                + np.sum(np.abs(sub_scale_pct - pop_scale_pct))
            ),
            4,
        ),
    }

    return selected_ids, stats
