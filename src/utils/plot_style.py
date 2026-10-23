"""Visualization styling for small object detection benchmarks.

Configures Matplotlib and Seaborn formatting for diagnostic figures:
- Single-axis horizontal gridlines
- Clean spine boundaries
- Typography hierarchy with Mathtext rendering
- Color palettes for scale bins and object categories
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Color palettes
COLORS = {
    "slate_dark": "#0F172A",
    "slate_body": "#1E293B",
    "slate_muted": "#64748B",
    "slate_light": "#94A3B8",
    "grid": "#E2E8F0",
    "background": "#FFFFFF",
    "indigo": "#4F46E5",
    "teal": "#0D9488",
    "amber": "#D97706",
    "rose": "#E11D48",
    "sky": "#0284C7",
    "emerald": "#059669",
    "violet": "#7C3AED",
    "coral": "#EA580C",
    "cyan": "#06B6D4",
}

# AI-TOD category palette
CATEGORY_PALETTE = {
    "airplane": "#4F46E5",
    "bridge": "#0D9488",
    "storage-tank": "#D97706",
    "ship": "#0284C7",
    "swimming-pool": "#06B6D4",
    "vehicle": "#E11D48",
    "person": "#8B5CF6",
    "wind-mill": "#059669",
}

# Scale bin palette
SCALE_BIN_PALETTE = {
    "vt": "#E11D48",  # Very Tiny: <= 8 px
    "t": "#F59E0B",   # Tiny: 8 < s <= 16 px
    "s": "#3B82F6",   # Small: 16 < s <= 32 px
    "m": "#10B981",   # Medium: 32 < s <= 64 px
    "l": "#64748B",   # Normal / Large: > 64 px
}

SCALE_BIN_LABELS = {
    "vt": r"$AP_{vt}$ ($\leq 8$ px)",
    "t": r"$AP_t$ ($8 - 16$ px)",
    "s": r"$AP_s$ ($16 - 32$ px)",
    "m": r"$AP_m$ ($32 - 64$ px)",
    "l": r"Normal / Large ($> 64$ px)",
}

COCO_COMPARISON_PALETTE = {
    "aitod_micro": "#E11D48",       # <= 16 px (VT + T)
    "aitod_coco_s": "#3B82F6",      # 16 - 32 px (S)
    "coco_medium_plus": "#94A3B8",  # > 32 px
}


def set_publication_style() -> None:
    """Set Matplotlib and Seaborn plotting parameters."""
    sns.set_theme(style="ticks", rc={"axes.facecolor": "#FFFFFF", "figure.facecolor": "#FFFFFF"})

    mpl.rcParams.update({
        # Typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "font.size": 10.0,
        "text.color": COLORS["slate_dark"],

        # Axes and labels
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.8,
        "axes.labelcolor": COLORS["slate_body"],
        "axes.labelsize": 10.5,
        "axes.labelweight": "normal",
        "axes.titlesize": 12.0,
        "axes.titleweight": "bold",
        "axes.titlepad": 12.0,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Horizontal gridlines
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,

        # Ticks
        "xtick.color": COLORS["slate_muted"],
        "ytick.color": COLORS["slate_muted"],
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,

        # Legend
        "legend.frameon": False,
        "legend.fontsize": 9.0,
        "legend.title_fontsize": 9.5,
        "legend.loc": "best",

        # Figure rendering
        "figure.titlesize": 13.5,
        "figure.titleweight": "bold",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "figure.facecolor": "#FFFFFF",
    })


def clean_spines(ax: mpl.axes.Axes, keep_left: bool = True, keep_bottom: bool = True) -> None:
    """Remove top and right spines, and format remaining spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(keep_left)
    ax.spines["bottom"].set_visible(keep_bottom)

    if keep_left:
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["left"].set_linewidth(0.8)
    if keep_bottom:
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.spines["bottom"].set_linewidth(0.8)


def add_header(
    fig: mpl.figure.Figure,
    title: str,
    subtitle: str | None = None,
    x: float = 0.05,
    y: float = 0.98,
) -> None:
    """Add title and subtitle header to figure."""
    fig.text(
        x,
        y,
        title,
        fontsize=13.0,
        fontweight="bold",
        color=COLORS["slate_dark"],
        ha="left",
        va="top",
    )
    if subtitle:
        fig.text(
            x,
            y - 0.038,
            subtitle,
            fontsize=9.5,
            fontweight="normal",
            color=COLORS["slate_muted"],
            ha="left",
            va="top",
        )


def save_figure(fig: mpl.figure.Figure, output_path: str | Path, dpi: int = 300, close: bool = True) -> Path:
    """Save figure with standard settings and create parent directories."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    if close:
        plt.close(fig)
    return path
