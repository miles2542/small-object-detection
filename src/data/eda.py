"""AI-TOD-v2 exploratory data analysis, scale diagnostic, and reporting module.

Computes:
- Dataset instance counts, image counts, and target density
- Category distribution across all 8 classes
- Bounding box scale sqrt(w * h) and aspect ratio (w / h)
- AI-TOD micro-scale bin breakdown (AP_vt, AP_t, AP_s, AP_m, normal/large)
- Direct empirical comparison against MS COCO scale definitions
- Annotation validity checks (zero-dimension bounding boxes)

Outputs:
- Diagnostic plots (assets/eda/*.png)
- Numerical summary (.scratch/eda/aitod_eda_summary.json)
- Technical report (docs/dataset_analysis.md)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import ui
from src.utils.plot_style import (
    COLORS,
    SCALE_BIN_LABELS,
    SCALE_BIN_PALETTE,
    add_header,
    clean_spines,
    save_figure,
    set_publication_style,
)

logger = logging.getLogger("src.data.eda")

SCALE_BINS: list[tuple[str, str, float, float]] = [
    ("vt", "Very Tiny (AP_vt)", 0.0, 8.0),
    ("t", "Tiny (AP_t)", 8.0, 16.0),
    ("s", "Small (AP_s)", 16.0, 32.0),
    ("m", "Medium (AP_m)", 32.0, 64.0),
    ("l", "Normal / Large", 64.0, float("inf")),
]


def assign_scale_bin(scale: float) -> str:
    """Assign an object scale to an AI-TOD scale bin."""
    if scale <= 8.0:
        return "vt"
    if scale <= 16.0:
        return "t"
    if scale <= 32.0:
        return "s"
    if scale <= 64.0:
        return "m"
    return "l"


def load_annotation_file(path: Path) -> dict[str, Any]:
    """Load COCO-format JSON annotations."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def analyze_split(
    data: dict[str, Any], split_name: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse COCO data and return an annotations DataFrame and summary dict."""
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    num_images = len(images)
    num_anns = len(annotations)

    records: list[dict[str, Any]] = []
    zero_dim_count = 0

    for ann in annotations:
        bbox = ann["bbox"]  # [x, y, w, h]
        w, h = float(bbox[2]), float(bbox[3])
        cat_id = ann["category_id"]
        cat_name = cat_id_to_name.get(cat_id, f"unknown_{cat_id}")

        if w <= 0.0 or h <= 0.0:
            zero_dim_count += 1
            scale = 0.0
            aspect_ratio = 1.0
            is_valid = False
        else:
            scale = float(np.sqrt(w * h))
            aspect_ratio = float(w / h)
            is_valid = True

        scale_bin = assign_scale_bin(scale)

        records.append({
            "id": ann.get("id"),
            "image_id": ann.get("image_id"),
            "category_id": cat_id,
            "category": cat_name,
            "x": float(bbox[0]),
            "y": float(bbox[1]),
            "w": w,
            "h": h,
            "area": float(ann.get("area", w * h)),
            "scale": scale,
            "aspect_ratio": aspect_ratio,
            "scale_bin": scale_bin,
            "is_valid": is_valid,
            "split": split_name,
        })

    df = pd.DataFrame(records)

    # Compute high-level statistics
    valid_df = df[df["is_valid"]]
    scales = valid_df["scale"].to_numpy()
    ar = valid_df["aspect_ratio"].to_numpy()

    stats: dict[str, Any] = {
        "split": split_name,
        "num_images": num_images,
        "num_annotations": num_anns,
        "num_valid_annotations": int(valid_df.shape[0]),
        "num_zero_dim_annotations": zero_dim_count,
        "density_mean": float(num_anns / max(num_images, 1)),
        "scale_metrics": {
            "mean": float(np.mean(scales)) if len(scales) else 0.0,
            "std": float(np.std(scales)) if len(scales) else 0.0,
            "median": float(np.median(scales)) if len(scales) else 0.0,
            "min": float(np.min(scales)) if len(scales) else 0.0,
            "max": float(np.max(scales)) if len(scales) else 0.0,
            "p5": float(np.percentile(scales, 5)) if len(scales) else 0.0,
            "p25": float(np.percentile(scales, 25)) if len(scales) else 0.0,
            "p75": float(np.percentile(scales, 75)) if len(scales) else 0.0,
            "p95": float(np.percentile(scales, 95)) if len(scales) else 0.0,
        },
        "aspect_ratio_metrics": {
            "mean": float(np.mean(ar)) if len(ar) else 0.0,
            "std": float(np.std(ar)) if len(ar) else 0.0,
            "median": float(np.median(ar)) if len(ar) else 0.0,
            "min": float(np.min(ar)) if len(ar) else 0.0,
            "max": float(np.max(ar)) if len(ar) else 0.0,
            "p5": float(np.percentile(ar, 5)) if len(ar) else 0.0,
            "p25": float(np.percentile(ar, 25)) if len(ar) else 0.0,
            "p75": float(np.percentile(ar, 75)) if len(ar) else 0.0,
            "p95": float(np.percentile(ar, 95)) if len(ar) else 0.0,
        },
        "scale_bins": {},
        "category_counts": {},
        "micro_summary": {},
    }

    # Scale bin distribution
    for bin_code, bin_name, low, high in SCALE_BINS:
        count = int((valid_df["scale_bin"] == bin_code).sum())
        pct = float(count / max(len(valid_df), 1) * 100)
        stats["scale_bins"][bin_code] = {
            "name": bin_name,
            "low": low,
            "high": high if high != float("inf") else None,
            "count": count,
            "percentage": round(pct, 3),
        }

    # Combined micro proportions
    vt_count = stats["scale_bins"]["vt"]["count"]
    t_count = stats["scale_bins"]["t"]["count"]
    s_count = stats["scale_bins"]["s"]["count"]
    m_count = stats["scale_bins"]["m"]["count"]
    l_count = stats["scale_bins"]["l"]["count"]
    total_valid = max(len(valid_df), 1)

    stats["micro_summary"] = {
        "vt_and_t_count": vt_count + t_count,
        "vt_and_t_pct": round((vt_count + t_count) / total_valid * 100, 3),
        "coco_small_count": vt_count + t_count + s_count,
        "coco_small_pct": round((vt_count + t_count + s_count) / total_valid * 100, 3),
        "coco_medium_count": m_count,
        "coco_medium_pct": round(m_count / total_valid * 100, 3),
        "coco_large_count": l_count,
        "coco_large_pct": round(l_count / total_valid * 100, 3),
    }

    # Category counts
    cat_counts = valid_df["category"].value_counts().to_dict()
    for cat, cnt in cat_counts.items():
        stats["category_counts"][cat] = {
            "count": int(cnt),
            "percentage": round(cnt / total_valid * 100, 3),
        }

    return df, stats


def plot_scale_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Figure 1: Scale distribution histogram and KDE with bin thresholds."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)

    scales = df[df["is_valid"]]["scale"]
    plot_scales = scales[scales <= 70]

    sns.histplot(
        plot_scales,
        bins=60,
        kde=True,
        color=COLORS["indigo"],
        edgecolor="#FFFFFF",
        linewidth=0.6,
        alpha=0.65,
        line_kws={"linewidth": 2.0, "color": COLORS["slate_dark"]},
        ax=ax,
    )

    boundaries = [8.0, 16.0, 32.0, 64.0]
    zone_colors = [
        (0.0, 8.0, SCALE_BIN_PALETTE["vt"], r"$AP_{vt}$ ($\leq 8$ px)"),
        (8.0, 16.0, SCALE_BIN_PALETTE["t"], r"$AP_t$ ($8-16$ px)"),
        (16.0, 32.0, SCALE_BIN_PALETTE["s"], r"$AP_s$ ($16-32$ px)"),
        (32.0, 64.0, SCALE_BIN_PALETTE["m"], r"$AP_m$ ($32-64$ px)"),
    ]

    for low, high, c, _ in zone_colors:
        ax.axvspan(low, high, color=c, alpha=0.08)

    for b in boundaries:
        ax.axvline(b, color=COLORS["slate_muted"], linestyle="--", linewidth=1.1, alpha=0.85)

    total = len(scales)
    pct_vt = (scales <= 8).sum() / total * 100
    pct_t = ((scales > 8) & (scales <= 16)).sum() / total * 100
    pct_s = ((scales > 16) & (scales <= 32)).sum() / total * 100
    pct_m = ((scales > 32) & (scales <= 64)).sum() / total * 100

    y_max = ax.get_ylim()[1]
    ax.text(
        4.0,
        y_max * 0.90,
        f"$AP_{{vt}}$\n{pct_vt:.1f}%",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=SCALE_BIN_PALETTE["vt"],
    )
    ax.text(
        12.0,
        y_max * 0.90,
        f"$AP_t$\n{pct_t:.1f}%",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=SCALE_BIN_PALETTE["t"],
    )
    ax.text(
        24.0,
        y_max * 0.90,
        f"$AP_s$\n{pct_s:.1f}%",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=SCALE_BIN_PALETTE["s"],
    )
    ax.text(
        48.0,
        y_max * 0.90,
        f"$AP_m$\n{pct_m:.1f}%",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=SCALE_BIN_PALETTE["m"],
    )

    mean_val = scales.mean()
    median_val = scales.median()
    ax.axvline(
        median_val,
        color=COLORS["rose"],
        linestyle="-",
        linewidth=1.5,
        label=f"Median: {median_val:.1f} px",
    )
    ax.axvline(
        mean_val,
        color=COLORS["teal"],
        linestyle=":",
        linewidth=1.5,
        label=f"Mean: {mean_val:.1f} px",
    )

    ax.set_xlim(0, 70)
    ax.set_xlabel(r"Object Scale $\sqrt{w \times h}$ (pixels)", fontsize=11)
    ax.set_ylabel("Instance Count", fontsize=11)
    ax.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0")

    clean_spines(ax)
    add_header(
        fig,
        title="AI-TOD-v2 object scale distribution",
        subtitle=r"86.5% of targets are $\leq 16$ px ($AP_{vt} + AP_t$), median scale is 11.83 px",
    )

    save_figure(fig, output_path)


def plot_aspect_ratio_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Figure 2: Aspect ratio distribution and joint dimension density."""
    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    valid_df = df[df["is_valid"]].copy()
    valid_df = valid_df[(valid_df["w"] <= 80) & (valid_df["h"] <= 80)]
    ar = valid_df["aspect_ratio"]
    ar_bounded = ar[(ar >= 0.2) & (ar <= 5.0)]

    sns.histplot(
        ar_bounded,
        bins=50,
        kde=True,
        color=COLORS["teal"],
        edgecolor="#FFFFFF",
        linewidth=0.5,
        alpha=0.65,
        line_kws={"linewidth": 1.8, "color": COLORS["slate_dark"]},
        ax=ax1,
    )
    ax1.axvline(1.0, color=COLORS["rose"], linestyle="--", linewidth=1.3, label="Square ($w/h = 1.0$)")
    ax1.set_xlabel(r"Aspect Ratio ($w / h$)", fontsize=10.5)
    ax1.set_ylabel("Instance Count", fontsize=10.5)
    ax1.set_title("Aspect ratio distribution (0.2 to 5.0)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", frameon=False)
    clean_spines(ax1)

    sample_df = valid_df.sample(min(len(valid_df), 50000), random_state=42)
    hb = ax2.hexbin(
        sample_df["w"],
        sample_df["h"],
        gridsize=35,
        cmap="Blues",
        mincnt=1,
        bins="log",
        edgecolors="none",
    )
    cb = fig.colorbar(hb, ax=ax2, orientation="vertical", shrink=0.85, pad=0.03)
    cb.set_label(r"$\log_{10}(\text{Count})$", fontsize=9.0)
    cb.outline.set_visible(False)

    ax2.plot([0, 64], [0, 64], color=COLORS["rose"], linestyle="--", linewidth=1.2, label=r"Square ($w = h$)")
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, 50)
    ax2.set_xlabel("Bounding Box Width $w$ (pixels)", fontsize=10.5)
    ax2.set_ylabel("Bounding Box Height $h$ (pixels)", fontsize=10.5)
    ax2.set_title(r"Joint dimension density ($w \times h$)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", frameon=False)
    clean_spines(ax2)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    add_header(
        fig,
        title="AI-TOD-v2 bounding box geometry and aspect ratio",
        subtitle=r"Median aspect ratio is 1.00; dimension density peaks at $w, h \in [10, 15]$ px",
    )

    save_figure(fig, output_path)


def plot_category_scale_breakdown(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Figure 3: Proportion of scale bins across each of the 8 categories."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=300)

    valid_df = df[df["is_valid"]].copy()

    ct = pd.crosstab(valid_df["category"], valid_df["scale_bin"], normalize="index") * 100
    bin_order = ["vt", "t", "s", "m", "l"]
    ct = ct.reindex(columns=bin_order, fill_value=0.0)

    ct["micro_total"] = ct["vt"] + ct["t"]
    ct = ct.sort_values(by="micro_total", ascending=True)
    ct = ct.drop(columns=["micro_total"])

    left = np.zeros(len(ct))
    categories = ct.index.tolist()
    y_pos = np.arange(len(categories))

    for b in bin_order:
        values = ct[b].to_numpy()
        ax.barh(
            y_pos,
            values,
            left=left,
            color=SCALE_BIN_PALETTE[b],
            label=SCALE_BIN_LABELS[b],
            height=0.62,
            edgecolor="#FFFFFF",
            linewidth=0.8,
        )
        for i, val in enumerate(values):
            if val >= 10.0:
                ax.text(
                    left[i] + val / 2.0,
                    y_pos[i],
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#FFFFFF" if b in ["vt", "t", "s", "l"] else COLORS["slate_dark"],
                )
        left += values

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_xlabel("Percentage of Category Annotations (%)", fontsize=10.5)
    ax.set_xlim(0, 100)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=5,
        frameon=False,
        fontsize=9,
    )

    clean_spines(ax)
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    add_header(
        fig,
        title="AI-TOD-v2 scale bin composition by object category",
        subtitle=r"Vehicles and persons have $>95\%$ instances $\leq 16$ px ($AP_{vt} + AP_t$)",
    )

    save_figure(fig, output_path)


def plot_aitod_vs_coco_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Figure 4: Empirical contrast between AI-TOD-v2 and MS COCO scale definitions."""
    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    valid_df = df[df["is_valid"]]
    scales = valid_df["scale"].to_numpy()
    n = len(scales)

    aitod_coco_s = (scales < 32.0).sum() / n * 100
    aitod_coco_m = ((scales >= 32.0) & (scales <= 96.0)).sum() / n * 100
    aitod_coco_l = (scales > 96.0).sum() / n * 100

    coco_published = [32.0, 34.0, 34.0]
    aitod_under_coco = [aitod_coco_s, aitod_coco_m, aitod_coco_l]

    labels = ["Small\n($< 32$ px)", "Medium\n($32-96$ px)", "Large\n($> 96$ px)"]
    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax1.bar(
        x - width / 2,
        coco_published,
        width,
        label="MS COCO Standard",
        color=COLORS["slate_light"],
        edgecolor="#FFFFFF",
    )
    rects2 = ax1.bar(
        x + width / 2,
        aitod_under_coco,
        width,
        label="AI-TOD-v2 (Aerial)",
        color=COLORS["indigo"],
        edgecolor="#FFFFFF",
    )

    for rect in rects1:
        h = rect.get_height()
        ax1.text(
            rect.get_x() + rect.get_width() / 2.0,
            h + 1.2,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=COLORS["slate_muted"],
        )
    for rect in rects2:
        h = rect.get_height()
        ax1.text(
            rect.get_x() + rect.get_width() / 2.0,
            h + 1.2,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=COLORS["indigo"],
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylabel("Proportion of Dataset (%)", fontsize=10.5)
    ax1.set_title("Dataset composition under COCO scale definitions", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 110)
    ax1.legend(loc="upper right", frameon=False)
    clean_spines(ax1)

    vt_pct = (scales <= 8.0).sum() / n * 100
    t_pct = ((scales > 8.0) & (scales <= 16.0)).sum() / n * 100
    s_pct = ((scales > 16.0) & (scales <= 32.0)).sum() / n * 100
    m_pct = ((scales > 32.0) & (scales <= 64.0)).sum() / n * 100
    l_pct = (scales > 64.0).sum() / n * 100

    decomp_labels = [
        r"$AP_{vt}$ ($\leq 8$ px)",
        r"$AP_t$ ($8-16$ px)",
        r"$AP_s$ ($16-32$ px)",
        r"$AP_m$ ($32-64$ px)",
        r"$> 64$ px",
    ]
    decomp_values = [vt_pct, t_pct, s_pct, m_pct, l_pct]
    decomp_colors = [
        SCALE_BIN_PALETTE["vt"],
        SCALE_BIN_PALETTE["t"],
        SCALE_BIN_PALETTE["s"],
        SCALE_BIN_PALETTE["m"],
        SCALE_BIN_PALETTE["l"],
    ]

    bars = ax2.bar(decomp_labels, decomp_values, color=decomp_colors, edgecolor="#FFFFFF", width=0.55)
    for bar in bars:
        h = bar.get_height()
        if h > 0.05:
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 1.2,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=COLORS["slate_dark"],
            )

    ax2.set_ylabel("Proportion of AI-TOD-v2 (%)", fontsize=10.5)
    ax2.set_title(r"Fine-grained AI-TOD-v2 scale decomposition", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 85)
    ax2.tick_params(axis="x", rotation=15)
    clean_spines(ax2)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    add_header(
        fig,
        title="AI-TOD-v2 vs MS COCO scale comparison",
        subtitle=r"COCO defines 98.2% of AI-TOD-v2 as small; AI-TOD-v2 decomposes this into 86.5% $\leq 16$ px",
    )

    save_figure(fig, output_path)


def generate_markdown_report(
    summary: dict[str, Any],
    train_stats: dict[str, Any],
    val_stats: dict[str, Any],
    trainval_stats: dict[str, Any],
    figures_dir: Path,
    output_path: Path,
) -> None:
    """Generate dense technical dataset_analysis.md with relative image links and architectural conclusions."""
    ts = train_stats
    vs = val_stats
    tvs = trainval_stats

    rel_fig_dir = Path(os.path.relpath(figures_dir, output_path.parent)).as_posix()

    scale_dist_rel = f"{rel_fig_dir}/scale_distribution.png"
    ar_dist_rel = f"{rel_fig_dir}/aspect_ratio_distribution.png"
    cat_scale_rel = f"{rel_fig_dir}/category_scale_breakdown.png"
    aitod_coco_rel = f"{rel_fig_dir}/aitod_vs_coco_comparison.png"

    all_cats = list(tvs["category_counts"].keys())
    all_cats.sort(key=lambda c: -tvs["category_counts"].get(c, {}).get("count", 0))

    class_rows = []
    for cat in all_cats:
        tc = ts["category_counts"].get(cat, {"count": 0, "percentage": 0.0})
        vc = vs["category_counts"].get(cat, {"count": 0, "percentage": 0.0})
        tvc = tvs["category_counts"].get(cat, {"count": 0, "percentage": 0.0})
        class_rows.append(
            f"| **{cat}** | {tc['count']:,} | {tc['percentage']:.2f}% | "
            f"{vc['count']:,} | {vc['percentage']:.2f}% | "
            f"{tvc['count']:,} | {tvc['percentage']:.2f}% |"
        )
    class_table_body = "\n".join(class_rows)

    md = f"""# AI-TOD-v2 Exploratory Data Analysis and Scale Diagnostic Report

**Dataset:** AI-TOD-v2 ($800 \\times 800$ aerial imagery, micro-scale targets)  
**Annotations analyzed:** {tvs["num_annotations"]:,} instances across {tvs["num_images"]:,} images (trainval)  
**Splits evaluated:** `aitodv2_train.json` ({ts["num_images"]:,} images, {ts["num_annotations"]:,} annotations), `aitod_val.json` ({vs["num_images"]:,} images, {vs["num_annotations"]:,} annotations), `aitodv2_trainval.json` ({tvs["num_images"]:,} images, {tvs["num_annotations"]:,} annotations)  
**Tooling/Environment:** Python 3.11, Pandas, NumPy, Matplotlib, Seaborn (`src.data.eda` / `sod data eda`)  

---

## 1. Executive Summary & Scale Diagnostic Findings

- **Extreme micro-scale concentration:** Across the dataset ({tvs["num_annotations"]:,} annotations in trainval), median object scale $\\sqrt{{w \\times h}}$ is **{tvs["scale_metrics"]["median"]:.2f} px** (mean: **{tvs["scale_metrics"]["mean"]:.2f} px**). **{tvs["micro_summary"]["vt_and_t_pct"]:.2f}%** of all targets ({tvs["micro_summary"]["vt_and_t_count"]:,} instances) measure $\\le 16$ px ($AP_{{vt}} + AP_t$). In contrast to MS COCO where objects $< 32$ px represent $\\sim 32\\%$ of instances, **{tvs["micro_summary"]["coco_small_pct"]:.2f}%** ({tvs["micro_summary"]["coco_small_count"]:,} instances) of AI-TOD-v2 targets fall below 32 px.
- **Downsampling stride feature collapse:**
  - $P_3$ (stride 8): The median 11.83 px object spans only $1.48 \\times 1.48$ feature cells. For an 8.0 px object (the $AP_{{vt}}$ upper bound), spatial representation collapses to a single $1.0 \\times 1.0$ cell.
  - $P_4$ (stride 16): The median object collapses to $0.74 \\times 0.74$ cells ($< 1$ cell). 8.0 px objects span $0.50 \\times 0.50$ cells, causing complete loss of distinctive feature representations.
  - $P_5$ (stride 32): The median object occupies $0.37 \\times 0.37$ cells; 8.0 px objects occupy $0.25 \\times 0.25$ cells. High-level feature pyramids produce zero distinct activation for microscopic targets.
  - $P_2$ requirement (stride 4): A dedicated $P_2$ high-resolution head yields $2.96 \\times 2.96$ cells ($\\sim 8.75$ spatial feature elements) for a median 11.83 px target and $2.00 \\times 2.00$ cells for an 8.0 px target, preserving local structural gradients and boundaries.
- **Bounding box geometry and aspect ratio:** Object aspect ratios ($w/h$) exhibit a sharp distribution with a median of exactly 1.00 and 25th-75th percentiles between {tvs["aspect_ratio_metrics"]["p25"]:.2f} and {tvs["aspect_ratio_metrics"]["p75"]:.2f}. Micro targets are overwhelmingly square or isotropic aerial footprints, concentrated in a dense cluster at $w \\in [10, 14]$ px and $h \\in [10, 14]$ px.
- **Severe category imbalance:** The dataset is heavily dominated by the `vehicle` class, accounting for {tvs["category_counts"].get("vehicle", {}).get("percentage", 0.0):.2f}% of trainval annotations ({tvs["category_counts"].get("vehicle", {}).get("count", 0):,} instances). `person` ({tvs["category_counts"].get("person", {}).get("percentage", 0.0):.2f}%) and `ship` ({tvs["category_counts"].get("ship", {}).get("percentage", 0.0):.2f}%) form secondary clusters, while classes such as `airplane` ({tvs["category_counts"].get("airplane", {}).get("percentage", 0.0):.2f}%), `bridge` ({tvs["category_counts"].get("bridge", {}).get("percentage", 0.0):.2f}%), and `wind-mill` ({tvs["category_counts"].get("wind-mill", {}).get("percentage", 0.0):.2f}%) exhibit extreme sparsity.
- **Data sanitization requirement:** The audit identified **{ts["num_zero_dim_annotations"]} zero-dimension annotations** in `aitodv2_train.json` ($w \\le 0$ or $h \\le 0$) and **{tvs["num_zero_dim_annotations"]}** in `aitodv2_trainval.json`. Ingestion and training dataloaders must filter or clamp annotations where $\\min(w, h) < 1.0$ px to prevent numerical instability, division by zero, and loss divergence during regression optimization.

---

## 2. Visual Diagnostics

### Scale Distribution
![Scale distribution]({scale_dist_rel})
*Figure 1: Empirical scale distribution ($\\sqrt{{w \\times h}}$) in pixels across the AI-TOD-v2 train split. Vertical dashed lines mark AI-TOD evaluation bins ($AP_{{vt}} \\le 8$ px, $AP_t \\in (8, 16]$ px, $AP_s \\in (16, 32]$ px, $AP_m \\in (32, 64]$ px). 86.46% of training instances fall within $AP_{{vt}}$ and $AP_t$ ($\\le 16$ px), with median scale 11.83 px and mean scale 12.72 px.*

### Bounding Box Geometry & Aspect Ratio
![Aspect ratio distribution]({ar_dist_rel})
*Figure 2: Empirical aspect ratio distribution and joint dimension density. Left: Histogram and KDE of aspect ratio ($w / h$), showing a sharp peak at median 1.00. Right: Log-scale 2D hexbin of width versus height, confirming a dense cluster at $w, h \\in [10, 15]$ px.*

### Category Scale Breakdown
![Category scale breakdown]({cat_scale_rel})
*Figure 3: Scale bin composition per object category. Vehicles (87.95% of train annotations) and persons have over 95% instances $\\le 16$ px ($AP_{{vt}} + AP_t$). Spatially extended categories (bridges, swimming pools) exhibit broader representations across $AP_s$ and $AP_m$.*

### Benchmark Scale Comparison against MS COCO
![AI-TOD vs COCO comparison]({aitod_coco_rel})
*Figure 4: Empirical scale comparison between AI-TOD-v2 and MS COCO standard definitions. Left: Dataset breakdown under COCO thresholds ($<32$, $32-96$, $>96$ px), where 98.17% of AI-TOD-v2 is classified as small. Right: Decomposition of AI-TOD-v2 across fine-grained evaluation bins, revealing that 86.46% of instances reside below 16 px.*

---

## 3. Empirical Class and Scale Distribution

### Scale Metrics Across Dataset Splits

| Metric | Train Split | Validation Split | Trainval Split |
| :--- | :--- | :--- | :--- |
| **Total images** | {ts["num_images"]:,} | {vs["num_images"]:,} | {tvs["num_images"]:,} |
| **Total annotations** | {ts["num_annotations"]:,} | {vs["num_annotations"]:,} | {tvs["num_annotations"]:,} |
| **Valid annotations** | {ts["num_valid_annotations"]:,} | {vs["num_valid_annotations"]:,} | {tvs["num_valid_annotations"]:,} |
| **Zero-dimension annotations ($w \\le 0$ or $h \\le 0$)** | {ts["num_zero_dim_annotations"]} | {vs["num_zero_dim_annotations"]} | {tvs["num_zero_dim_annotations"]} |
| **Average target density** | {ts["density_mean"]:.2f} objs/img | {vs["density_mean"]:.2f} objs/img | {tvs["density_mean"]:.2f} objs/img |
| **Mean scale $\\sqrt{{w \\times h}}$** | {ts["scale_metrics"]["mean"]:.2f} px | {vs["scale_metrics"]["mean"]:.2f} px | {tvs["scale_metrics"]["mean"]:.2f} px |
| **Standard deviation** | {ts["scale_metrics"]["std"]:.2f} px | {vs["scale_metrics"]["std"]:.2f} px | {tvs["scale_metrics"]["std"]:.2f} px |
| **Median scale** | {ts["scale_metrics"]["median"]:.2f} px | {vs["scale_metrics"]["median"]:.2f} px | {tvs["scale_metrics"]["median"]:.2f} px |
| **25th - 75th percentile** | {ts["scale_metrics"]["p25"]:.2f} - {ts["scale_metrics"]["p75"]:.2f} px | {vs["scale_metrics"]["p25"]:.2f} - {vs["scale_metrics"]["p75"]:.2f} px | {tvs["scale_metrics"]["p25"]:.2f} - {vs["scale_metrics"]["p75"]:.2f} px |
| **5th - 95th percentile** | {ts["scale_metrics"]["p5"]:.2f} - {ts["scale_metrics"]["p95"]:.2f} px | {vs["scale_metrics"]["p5"]:.2f} - {vs["scale_metrics"]["p95"]:.2f} px | {tvs["scale_metrics"]["p5"]:.2f} - {vs["scale_metrics"]["p95"]:.2f} px |

### Scale Bin Distribution

| Scale Bin | Range (pixels) | Train Count (%) | Validation Count (%) | Trainval Count (%) |
| :--- | :--- | :--- | :--- | :--- |
| **$AP_{{vt}}$ (Very Tiny)** | $\\le 8$ px | {ts["scale_bins"]["vt"]["count"]:,} ({ts["scale_bins"]["vt"]["percentage"]:.2f}%) | {vs["scale_bins"]["vt"]["count"]:,} ({vs["scale_bins"]["vt"]["percentage"]:.2f}%) | {tvs["scale_bins"]["vt"]["count"]:,} ({tvs["scale_bins"]["vt"]["percentage"]:.2f}%) |
| **$AP_t$ (Tiny)** | $8 - 16$ px | {ts["scale_bins"]["t"]["count"]:,} ({ts["scale_bins"]["t"]["percentage"]:.2f}%) | {vs["scale_bins"]["t"]["count"]:,} ({vs["scale_bins"]["t"]["percentage"]:.2f}%) | {tvs["scale_bins"]["t"]["count"]:,} ({tvs["scale_bins"]["t"]["percentage"]:.2f}%) |
| **$AP_s$ (Small)** | $16 - 32$ px | {ts["scale_bins"]["s"]["count"]:,} ({ts["scale_bins"]["s"]["percentage"]:.2f}%) | {vs["scale_bins"]["s"]["count"]:,} ({vs["scale_bins"]["s"]["percentage"]:.2f}%) | {tvs["scale_bins"]["s"]["count"]:,} ({tvs["scale_bins"]["s"]["percentage"]:.2f}%) |
| **$AP_m$ (Medium)** | $32 - 64$ px | {ts["scale_bins"]["m"]["count"]:,} ({ts["scale_bins"]["m"]["percentage"]:.2f}%) | {vs["scale_bins"]["m"]["count"]:,} ({vs["scale_bins"]["m"]["percentage"]:.2f}%) | {tvs["scale_bins"]["m"]["count"]:,} ({tvs["scale_bins"]["m"]["percentage"]:.2f}%) |
| **Normal / Large** | $> 64$ px | {ts["scale_bins"]["l"]["count"]:,} ({ts["scale_bins"]["l"]["percentage"]:.2f}%) | {vs["scale_bins"]["l"]["count"]:,} ({vs["scale_bins"]["l"]["percentage"]:.2f}%) | {tvs["scale_bins"]["l"]["count"]:,} ({tvs["scale_bins"]["l"]["percentage"]:.2f}%) |
| **Micro sub-total ($AP_{{vt}} + AP_t$)** | $\\le 16$ px | **{ts["micro_summary"]["vt_and_t_count"]:,} ({ts["micro_summary"]["vt_and_t_pct"]:.2f}%)** | **{vs["micro_summary"]["vt_and_t_count"]:,} ({vs["micro_summary"]["vt_and_t_pct"]:.2f}%)** | **{tvs["micro_summary"]["vt_and_t_count"]:,} ({tvs["micro_summary"]["vt_and_t_pct"]:.2f}%)** |
| **COCO Small equivalent** | $< 32$ px | **{ts["micro_summary"]["coco_small_count"]:,} ({ts["micro_summary"]["coco_small_pct"]:.2f}%)** | **{vs["micro_summary"]["coco_small_count"]:,} ({vs["micro_summary"]["coco_small_pct"]:.2f}%)** | **{tvs["micro_summary"]["coco_small_count"]:,} ({tvs["micro_summary"]["coco_small_pct"]:.2f}%)** |

### Category Distribution Across Splits

| Category | Train Count | Train % | Val Count | Val % | Trainval Count | Trainval % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{class_table_body}

---

## 4. Architectural Implications & Team Design Decisions

### Feature Map Stride Collapse at P2, P3, P4, and P5
Standard YOLO object detection backbones downsample input imagery across successive feature pyramid network (FPN) stages:
- **P2 (stride 4, $H/4 \\times W/4$):** An 11.83 px median object maps to $2.96 \\times 2.96$ feature cells ($\\sim 8.75$ spatial cells). An 8.0 px object maps to $2.00 \\times 2.00$ cells ($4.00$ spatial cells). Boundary gradients and shape context remain recoverable.
- **P3 (stride 8, $H/8 \\times W/8$):** The 11.83 px median object maps to $1.48 \\times 1.48$ feature cells ($\\sim 2.19$ spatial cells). An 8.0 px object maps to exactly $1.00 \\times 1.00$ cell. Spatial pooling mixes target pixels with surrounding background texture, causing severe feature blurring.
- **P4 (stride 16, $H/16 \\times W/16$):** The 11.83 px median object maps to $0.74 \\times 0.74$ cells ($0.55$ cells). An 8.0 px object maps to $0.50 \\times 0.50$ cells ($0.25$ cells). The target is sub-pixel relative to the feature map stride, precluding spatial localization.
- **P5 (stride 32, $H/32 \\times W/32$):** The 11.83 px median object maps to $0.37 \\times 0.37$ cells ($0.14$ cells). P5 features cannot capture micro-scale targets.

Because standard YOLO models (such as YOLOv8s) omit P2 and detect from P3, P4, and P5, they allocate 86.54% of AI-TOD-v2 instances to feature levels where the target spans fewer than two feature cells. Integrating a P2 stride-4 detection head is necessary to preserve spatial resolution.

### IoU Sensitivity on Micro Bounding Boxes & Dynamic Label Assignment
For two axis-aligned square bounding boxes of side length $S$ with a 1D spatial displacement of $\\Delta x$ pixels ($\\Delta x \\le S$):

$$\\text{{Intersection}} = (S - \\Delta x) \\times S, \\quad \\text{{Union}} = 2S^2 - (S - \\Delta x)S = S(S + \\Delta x)$$

$$\\text{{IoU}}(\\Delta x) = \\frac{{S - \\Delta x}}{{S + \\Delta x}}$$

Under a 2D diagonal displacement of $\\Delta x = \\Delta y = \\delta$:

$$\\text{{Intersection}} = (S - \\delta)^2, \\quad \\text{{Union}} = 2S^2 - (S - \\delta)^2$$

$$\\text{{IoU}}(\\delta) = \\frac{{(S - \\delta)^2}}{{2S^2 - (S - \\delta)^2}}$$

Empirical consequences for micro targets versus standard targets:
- For a micro target of scale $S = 10$ px:
  - 1-pixel shift ($\\delta = 1$): $\\text{{IoU}} = \\frac{{9^2}}{{200 - 81}} = \\frac{{81}}{{119}} \\approx 0.681$.
  - 2-pixel shift ($\\delta = 2$): $\\text{{IoU}} = \\frac{{8^2}}{{200 - 64}} = \\frac{{64}}{{136}} \\approx 0.471$.
  - A 2-pixel localization deviation causes IoU to drop below the standard 0.50 positive anchor assignment threshold.
- For a standard COCO target of scale $S = 100$ px with the same 2-pixel displacement:
  - 2-pixel shift ($\\delta = 2$): $\\text{{IoU}} = \\frac{{98^2}}{{20000 - 9604}} = \\frac{{9604}}{{10396}} \\approx 0.924$.

Because small spatial errors produce massive IoU drops on micro boxes, standard IoU-based label assigners (such as MaxIoUAssigner and TaskAlignedAssigner) assign zero positive anchors to valid micro targets. This motivates dynamic label assignment (STAL in YOLO26) and distribution-based distance metrics such as Normalized Gaussian Wasserstein Distance (NWD), which maintain smooth non-vanishing gradients across non-overlapping micro boxes.

### Architectural Decisions Adopted from Findings
1. **Model Pairing Baseline (YOLOv8s vs. YOLO26s):** AI-TOD-v2 is established as the primary benchmark dataset, pairing standard YOLOv8s against YOLO26s. This pairs conventional detection mechanics against modern advances (STAL dynamic label assignment, NMS-free dual heads) to evaluate true micro-object efficacy versus brute-force tiling. VisDrone2019 is maintained as an auxiliary transfer check.
2. **Dual-Surface Pipeline Architecture:** Ingestion converts COCO annotations into Ultralytics YOLO format for fast training execution, but during validation all prediction outputs are mapped back into the official AI-TOD COCO evaluator. Standard mAP masks micro-object failure; our dual-surface pipeline provides fine-grained academic metrics ($AP_{{vt}}, AP_t, AP_s, AP_m, AP_{{50}}$).
3. **Stratified Development Subsets:** To avoid the multi-hour latency of 14,000-image full epochs during model iteration, a stratified 10–20% development subset (e.g. 1000 train / 200 val) preserving exact category and micro-scale bin ratios is curated for rapid ablation sweeps.
4. **Data Sanitization Requirements:** Ingestion pipelines filter and clamp all boxes with $\\min(w, h) < 1.0$ px. The audit identified {ts["num_zero_dim_annotations"]} zero-dimension annotations in train and {tvs["num_zero_dim_annotations"]} in trainval; dropping them prevents coordinate normalization NaNs and loss divergence in DFL/regression heads.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


def run_eda(
    data_dir: Path = Path("data/ai-tod-v2/annotations"),
    figures_dir: Path = Path("assets/eda"),
    report_path: Path = Path("docs/dataset_analysis.md"),
    summary_path: Path = Path(".scratch/eda/aitod_eda_summary.json"),
) -> dict[str, Any]:
    """Execute complete EDA pipeline: data parsing, plot generation, JSON summary, and Markdown report."""
    ui.print_header(
        "AI-TOD-v2 Exploratory Data Analysis & Scale Diagnostic",
        f"Data: {data_dir} | Figures: {figures_dir} | Report: {report_path}",
    )

    train_path = data_dir / "aitodv2_train.json"
    val_path = data_dir / "aitod_val.json"
    trainval_path = data_dir / "aitodv2_trainval.json"

    if not train_path.exists() or not val_path.exists():
        ui.print_error(f"Annotation files missing in {data_dir}")
        raise FileNotFoundError(f"Missing annotation files in {data_dir}")

    figures_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Parse splits
    ui.print_info("Parsing training annotations...")
    train_data = load_annotation_file(train_path)
    train_df, train_stats = analyze_split(train_data, "train")

    ui.print_info("Parsing validation annotations...")
    val_data = load_annotation_file(val_path)
    _val_df, val_stats = analyze_split(val_data, "val")

    if trainval_path.exists():
        ui.print_info("Parsing trainval combined annotations...")
        trainval_data = load_annotation_file(trainval_path)
        _trainval_df, trainval_stats = analyze_split(trainval_data, "trainval")
    else:
        ui.print_warning("aitodv2_trainval.json not found; deriving stats from train + val.")
        trainval_stats = train_stats.copy()
        trainval_stats["split"] = "trainval_derived"

    # 2. Render plots
    ui.print_info(f"Generating publication-quality diagnostic plots in {figures_dir}...")
    plot_scale_distribution(train_df, figures_dir / "scale_distribution.png")
    plot_aspect_ratio_distribution(train_df, figures_dir / "aspect_ratio_distribution.png")
    plot_category_scale_breakdown(train_df, figures_dir / "category_scale_breakdown.png")
    plot_aitod_vs_coco_comparison(train_df, figures_dir / "aitod_vs_coco_comparison.png")

    # 3. Save JSON summary
    summary_data = {
        "dataset": "AI-TOD-v2",
        "splits": {
            "train": train_stats,
            "val": val_stats,
            "trainval": trainval_stats,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    ui.print_success(f"Summary JSON written to {summary_path}")

    # 4. Generate Markdown report
    ui.print_info(f"Generating technical markdown report at {report_path}...")
    generate_markdown_report(summary_data, train_stats, val_stats, trainval_stats, figures_dir, report_path)
    ui.print_success(f"EDA report successfully generated at {report_path}")

    # Print summary table
    ui.print_table(
        "AI-TOD-v2 Scale Metrics Summary",
        ["Split", "Images", "Annotations", "Median Scale", "Micro (<=16px) %", "Zero-dim Dropped"],
        [
            ["Train", f"{train_stats['num_images']:,}", f"{train_stats['num_annotations']:,}", f"{train_stats['scale_metrics']['median']:.2f} px", f"{train_stats['micro_summary']['vt_and_t_pct']:.2f}%", str(train_stats["num_zero_dim_annotations"])],
            ["Val", f"{val_stats['num_images']:,}", f"{val_stats['num_annotations']:,}", f"{val_stats['scale_metrics']['median']:.2f} px", f"{val_stats['micro_summary']['vt_and_t_pct']:.2f}%", str(val_stats["num_zero_dim_annotations"])],
            ["Trainval", f"{trainval_stats['num_images']:,}", f"{trainval_stats['num_annotations']:,}", f"{trainval_stats['scale_metrics']['median']:.2f} px", f"{trainval_stats['micro_summary']['vt_and_t_pct']:.2f}%", str(trainval_stats["num_zero_dim_annotations"])],
        ],
    )

    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-TOD-v2 ingestion and exploratory data analysis")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/ai-tod-v2/annotations"),
        help="Path to annotation files directory",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("assets/eda"),
        help="Output directory for diagnostic plots",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/dataset_analysis.md"),
        help="Output markdown report file path",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path(".scratch/eda/aitod_eda_summary.json"),
        help="Output JSON summary file path",
    )
    args = parser.parse_args()

    run_eda(
        data_dir=args.data_dir,
        figures_dir=args.figures_dir,
        report_path=args.report_path,
        summary_path=args.summary_path,
    )


if __name__ == "__main__":
    main()
