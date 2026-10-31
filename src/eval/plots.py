"""Saved-prediction previews using the project's publication style."""

from __future__ import annotations

import json
from pathlib import Path


def plot_input_density(annotations: dict[str, Path], output: Path) -> dict:
    """Measure GT density and an idealized global-cap ceiling, without model predictions."""
    from collections import Counter

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from src.utils.plot_style import (
        COLORS,
        add_header,
        clean_spines,
        save_figure,
        set_publication_style,
    )

    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(top=0.79, bottom=0.17, wspace=0.28)
    add_header(
        fig,
        "GT density: why a global output cap can matter",
        "Valid boxes only | Idealized cap ceiling assumes one correct output per GT; no measured model recall",
    )
    summary = {}
    for (name, path), color in zip(annotations.items(), [COLORS["indigo"], COLORS["teal"]]):
        coco = json.loads(path.read_text(encoding="utf-8"))
        counts = Counter(
            a["image_id"]
            for a in coco["annotations"]
            if a["bbox"][2] > 0 and a["bbox"][3] > 0 and a["area"] > 0
        )
        values = np.array([counts[i["id"]] for i in coco["images"]])
        sorted_values = np.sort(values)
        axes[0].step(
            sorted_values,
            100 * np.arange(1, len(values) + 1) / len(values),
            where="post",
            color=color,
            label=f"{name}, n={len(values):,}",
        )
        caps = np.array([100, 300, 500, 1000, 1500])
        ceilings = [100 * np.minimum(values, cap).sum() / values.sum() for cap in caps]
        axes[1].plot(caps, ceilings, marker="o", color=color, label=name)
        summary[name] = {
            "images": len(values),
            "valid_gt": int(values.sum()),
            "max_gt": int(values.max()),
            "zero_gt_images": int((values == 0).sum()),
            "caps": {
                str(cap): {
                    "images_above": int((values > cap).sum()),
                    "unavoidable_misses": int(np.maximum(values - cap, 0).sum()),
                    "ideal_recall_ceiling_percent": ceiling,
                }
                for cap, ceiling in zip(caps.tolist(), ceilings)
            },
        }
    axes[0].set(
        xlabel="Eligible GT per image (log scale above 1)",
        ylabel="Images at or below count (%)",
        ylim=(0, 102),
    )
    axes[0].set_xscale("symlog", linthresh=1, linscale=0.6)
    axes[0].set_xlim(0, 1.1 * max(item["max_gt"] for item in summary.values()))
    axes[1].set(xlabel="Global detections per image cap", ylabel="Ideal pooled recall ceiling (%)")
    for ax in axes:
        ax.legend(fontsize=8)
        clean_spines(ax)
    save_figure(fig, output)
    return summary


def plot_area_membership(output: Path) -> None:
    """Analytic exact-boundary membership makes overlap and out-of-bin objects explicit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.colors import ListedColormap

    from src.eval.protocol import evaluation_protocol
    from src.utils.plot_style import add_header, save_figure, set_publication_style

    set_publication_style()
    protocol = evaluation_protocol("assignment")
    areas = [3.99, 4, 63.99, 64, 64.01, 256, 1024, 4096, 4096.01]
    matrix = np.array(
        [[int(low <= a <= high) for a in areas] for low, high in protocol["area_ranges"]]
    )
    fig, ax = plt.subplots(figsize=(12, 4.4))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.76, bottom=0.20)
    add_header(
        fig,
        "Area-range membership: shared endpoints are intentional",
        "Analytic examples, not measured frequencies | Assignment profile | 1=included; 0=excluded",
    )
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=ListedColormap(["#EDF2F7", "#5EBEB0"]),
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=1.5,
        linecolor="white",
        ax=ax,
        xticklabels=[f"{a:g}" for a in areas],
        yticklabels=protocol["area_labels"],
    )
    ax.set_xlabel("Stored annotation area (px²); columns are separate cases, not a continuous axis")
    ax.set_ylabel("Evaluation group")
    ax.tick_params(axis="y", labelrotation=0)
    save_figure(fig, output)


def plot_export_diagnostics(
    runs: dict[str, Path], annotations: Path, output: Path, *, scores: bool = False
) -> dict:
    """Manifest-based coverage or score-retention diagnostics; no TP/FP claims without matching."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from src.utils.plot_style import (
        COLORS,
        add_header,
        clean_spines,
        save_figure,
        set_publication_style,
    )

    set_publication_style()
    coco = json.loads(annotations.read_text(encoding="utf-8"))
    manifests = {
        name: json.loads((path / "manifest.json").read_text()) for name, path in runs.items()
    }
    ids = next(iter(manifests.values()))["image_ids"]
    if any(m["image_ids"] != ids or m["status"] != "complete" for m in manifests.values()):
        raise ValueError(
            "Export diagnostics require complete runs with identical ordered image IDs"
        )
    selected = set(ids)
    eligible = [
        a
        for a in coco["annotations"]
        if a["image_id"] in selected and a["bbox"][2] > 0 and a["bbox"][3] > 0 and a["area"] > 0
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(top=0.78, bottom=0.18, wspace=0.28)
    add_header(
        fig,
        "Export score retention and postprocessing sensitivity"
        if scores
        else "Export coverage: output volume and silent-image risk",
        f"Pretrained smoke only | Output counts are not true positives | Same {len(ids)} validation IDs",
    )
    summary = {}
    for (name, path), color, style in zip(
        runs.items(), [COLORS["indigo"], COLORS["teal"], COLORS["amber"]], ["-", "-", "--"]
    ):
        m = manifests[name]
        from src.eval.audit import sha256

        if (
            sha256(annotations) != m["annotations"]["sha256"]
            or sha256(path / "predictions.json") != m["predictions_sha256"]
        ):
            raise ValueError("Export diagnostics input fingerprint mismatch")
        rows = json.loads((path / "predictions.json").read_text())
        counts = np.array([m["prediction_counts"][str(i)] for i in ids])
        empty = set(m["zero_detection_image_ids"])
        hidden = [a for a in eligible if a["image_id"] in empty]
        mapped = {entry["category_id"] for entry in m["category_mapping"].values()}
        thresholds = [0.001, 0.01, 0.1, 0.25, 0.5]
        retention = [sum(r["score"] >= threshold for r in rows) for threshold in thresholds]
        summary[name] = {
            "detections": len(rows),
            "empty_images": len(empty),
            "all_gt_in_empty_images": len(hidden),
            "shared_class_gt_in_empty_images": sum(a["category_id"] in mapped for a in hidden),
            "count_percentiles": {
                str(q): float(np.percentile(counts, q)) for q in [25, 50, 75, 95]
            },
            "retained_by_score": dict(zip(map(str, thresholds), retention)),
        }
        if scores:
            axes[0].plot(
                thresholds,
                100 * np.array(retention) / max(1, len(rows)),
                marker="o",
                color=color,
                linestyle=style,
                label=name,
            )
            axes[1].plot(
                thresholds, retention, marker="o", color=color, linestyle=style, label=name
            )
        else:
            axes[0].step(
                np.sort(counts),
                100 * np.arange(1, len(counts) + 1) / len(counts),
                where="post",
                color=color,
                label=name,
            )
            x = len(summary) - 1
            axes[1].bar(x - 0.18, len(hidden), width=0.34, color=color)
            axes[1].bar(
                x + 0.18,
                sum(a["category_id"] in mapped for a in hidden),
                width=0.34,
                color=color,
                alpha=0.4,
            )
            for offset, value in [
                (-0.18, len(hidden)),
                (0.18, summary[name]["shared_class_gt_in_empty_images"]),
            ]:
                axes[1].text(
                    x + offset, value + 3, str(value), ha="center", fontsize=9, fontweight="bold"
                )
    if scores:
        for ax, ylabel in zip(
            axes, ["Exported boxes retained (%)", "Exported boxes retained (count)"]
        ):
            ax.set(xscale="log", xlabel="Confidence cutoff (log scale)", ylabel=ylabel)
            ax.legend(fontsize=8)
        axes[1].set_yscale("log")
    else:
        axes[0].set(xlabel="Exported detections per image", ylabel="Images at or below count (%)")
        axes[0].legend(fontsize=8)
        axes[1].set_xticks(range(len(runs)), list(runs))
        axes[1].set(
            ylabel="Eligible GT on zero-output images",
            title="Dark=all classes; pale=shared output classes",
        )
        axes[1].margins(y=0.18)
    for ax in axes:
        clean_spines(ax)
    save_figure(fig, output)
    return summary


def plot_export_preview(
    annotations: Path, image_root: Path, runs: dict[str, Path], output: Path
) -> None:
    """Show a common image and crop; overlays inspect coordinates, not detection accuracy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from PIL import Image

    from src.utils.plot_style import (
        COLORS,
        add_header,
        clean_spines,
        save_figure,
        set_publication_style,
    )

    coco = json.loads(annotations.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in coco["images"]}
    predictions = {
        name: json.loads((path / "predictions.json").read_text()) for name, path in runs.items()
    }
    combined = [row for rows in predictions.values() for row in rows]
    if not combined:
        raise ValueError("No predictions to preview; empty exports remain valid")
    anchor = max(combined, key=lambda row: row["score"])
    iid = anchor["image_id"]
    x, y, w, h = anchor["bbox"]
    cx, cy, half = x + w / 2, y + h / 2, max(80, w, h)
    left = max(0, min(cx - half, images[iid]["width"] - 2 * half))
    top = max(0, min(cy - half, images[iid]["height"] - 2 * half))
    right, bottom = (
        min(images[iid]["width"], left + 2 * half),
        min(images[iid]["height"], top + 2 * half),
    )
    set_publication_style()
    fig, axes = plt.subplots(len(runs), 2, figsize=(10, 4 * len(runs)), squeeze=False)
    fig.subplots_adjust(top=0.84, bottom=0.09, wspace=0.18, hspace=0.30)
    add_header(
        fig,
        "Prediction export: original-pixel overlay check",
        f"Pretrained smoke only | COCO image ID {iid} | {images[iid]['file_name']}",
    )
    with Image.open(image_root / images[iid]["file_name"]) as source:
        pixels = source.convert("RGB")
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    for row, (name, rows) in enumerate(predictions.items()):
        manifest = json.loads((runs[name] / "manifest.json").read_text())
        supported = {entry["category_id"] for entry in manifest["category_mapping"].values()}
        gt = [
            a for a in coco["annotations"] if a["image_id"] == iid and a["category_id"] in supported
        ]
        shown = sorted([p for p in rows if p["image_id"] == iid], key=lambda p: -p["score"])[:8]
        for col, ax in enumerate(axes[row]):
            ax.imshow(pixels)
            for item in gt:
                bx, by, bw, bh = item["bbox"]
                ax.add_patch(
                    Rectangle(
                        (bx, by),
                        bw,
                        bh,
                        fill=False,
                        edgecolor=COLORS["teal"],
                        linewidth=1,
                        linestyle="--",
                    )
                )
            for item in shown:
                bx, by, bw, bh = item["bbox"]
                ax.add_patch(
                    Rectangle((bx, by), bw, bh, fill=False, edgecolor=COLORS["rose"], linewidth=1)
                )
                if col and left <= bx <= right and top <= by <= bottom:
                    ax.text(
                        bx,
                        by,
                        f"{categories[item['category_id']]} {item['score']:.3f}",
                        fontsize=7,
                        color=COLORS["rose"],
                        ha="right" if bx > (left + right) / 2 else "left",
                        bbox={"facecolor": "white", "alpha": 0.8, "pad": 1},
                    )
            if col:
                ax.set_xlim(left, right)
                ax.set_ylim(bottom, top)
            ax.set_title(
                f"{name} | {'Zoom in original coordinates' if col else 'Top 8 predictions + supported GT'}"
            )
            ax.grid(False)
            ax.set_xlabel("Original x (px)")
            ax.set_ylabel("Original y (px)")
            clean_spines(ax)
    fig.legend(
        handles=[
            Patch(color=COLORS["teal"], label="Ground truth (shared classes)"),
            Patch(color=COLORS["rose"], label="Exported prediction"),
        ],
        loc="lower center",
        ncol=2,
    )
    save_figure(fig, output)


def plot_recall_iou(runs: dict[str, Path], output: Path, category_id: int = 6) -> None:
    """Compare localization-sensitive maximum recall without re-running inference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from src.utils.plot_style import (
        SCALE_BIN_PALETTE,
        add_header,
        clean_spines,
        save_figure,
        set_publication_style,
    )

    set_publication_style()
    fig, axes = plt.subplots(1, len(runs), figsize=(11, 5), squeeze=False, sharey=True)
    fig.subplots_adjust(top=0.77, bottom=0.18, wspace=0.15)
    add_header(
        fig,
        "Person recall versus localization requirement",
        "Smoke only | Maximum recall from scores >= 0.001 | Dotted lines denote fewer than five GT",
    )
    for ax, (name, path) in zip(axes.flat, runs.items()):
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        with np.load(path / "curves.npz") as data:
            k = int(np.flatnonzero(data["category_ids"] == category_id)[0])
            for label, code in zip(["verytiny", "tiny", "small", "medium"], ["vt", "t", "s", "m"]):
                a = list(data["area_labels"]).index(label)
                support = metrics["per_class"][str(category_id)]["bins"][label]["support"]
                values = data["recall"][:, k, a, -1]
                ax.plot(
                    data["iou_thresholds"],
                    np.where(values >= 0, values * 100, np.nan),
                    marker="o",
                    markersize=3,
                    color=SCALE_BIN_PALETTE[code],
                    label=f"{code}, n={support}",
                    linestyle=":" if support < 5 else "-",
                )
        ax.set(title=name, xlabel="Required IoU", ylabel="Maximum recall (%)", ylim=(0, 105))
        ax.legend()
        clean_spines(ax)
    save_figure(fig, output)


def plot_confidence_tradeoff(diagnostics: Path, output: Path) -> None:
    """Plot raw-count operating precision/recall, not interpolated COCO PR values."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    from src.utils.plot_style import (
        COLORS,
        add_header,
        clean_spines,
        save_figure,
        set_publication_style,
    )

    data = json.loads(diagnostics.read_text(encoding="utf-8"))
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(top=0.77, bottom=0.19, wspace=0.23)
    support = next(iter(data["models"].values()))["support"]
    add_header(
        fig,
        "Person confidence threshold: retained precision versus recall",
        f"Smoke only | IoU 0.50 | n={support} GT | No retained predictions means precision N/A",
    )
    for (name, values), color in zip(data["models"].items(), [COLORS["indigo"], COLORS["teal"]]):
        points = values["points"]
        for ax, metric in zip(axes, ["precision", "recall"]):
            ax.plot(
                [point["score"] for point in points],
                [np.nan if point[metric] is None else 100 * point[metric] for point in points],
                marker="o",
                markersize=4,
                color=color,
                label=name,
            )
    for ax, metric in zip(axes, ["Precision", "Recall"]):
        ax.set(
            xscale="log",
            xlabel="Confidence cutoff (log scale)",
            ylabel=f"{metric} (%)",
            ylim=(0, 105),
        )
        ax.set_xticks([0.001, 0.01, 0.1, 0.25, 0.9])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.axvline(0.25, linestyle="--", color=COLORS["slate_muted"], linewidth=0.9)
        ax.legend()
        clean_spines(ax)
    save_figure(fig, output)


def plot_cli_workflow(output: Path) -> None:
    """Show the two CLI entry paths and their shared validation/evaluation surface."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    from src.utils.plot_style import COLORS, add_header, save_figure, set_publication_style

    set_publication_style()
    fig, ax = plt.subplots(figsize=(13, 4.8))
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.045, top=0.80)
    add_header(
        fig,
        "Evaluation CLI: two entry paths, one evaluator",
        "Saved JSON reuses inference | Checkpoint mode selects images and exports original IDs",
        x=0.035,
        y=0.98,
    )
    ax.set(xlim=(0, 12), ylim=(0, 3.8))
    ax.axis("off")
    boxes = [
        (0.1, 2.35, "Saved predictions", "--predictions\n--data COCO JSON", "#DBEAFE", "#1D4ED8"),
        (0.1, 0.15, "Checkpoint input", "--weights\n--data YAML or JSON", "#EDE9FE", "#6D28D9"),
        (
            4.25,
            2.35,
            "Validate and fingerprint",
            "Original image / class IDs\nManifest checks when present",
            "#CFFAFE",
            "#0E7490",
        ),
        (
            4.25,
            0.15,
            "Prepare and export",
            "Exact image set; normalized GT\nOriginal-ID prediction JSON",
            "#FEF3C7",
            "#92400E",
        ),
        (
            8.4,
            2.35,
            "Shared COCO evaluator",
            "Fixed size and IoU protocol\nSame matching semantics",
            "#D1FAE5",
            "#047857",
        ),
        (
            8.4,
            0.15,
            "Metrics and evidence",
            "AP / AR; confidence counts\nJSON + PR arrays + tables",
            "#FFE4E6",
            "#BE123C",
        ),
    ]
    for x, y, title, body, fill, tone in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                3.5,
                1.3,
                boxstyle="round,pad=0.06,rounding_size=0.10",
                facecolor=fill,
                edgecolor=tone,
                linewidth=1.5,
            )
        )
        ax.text(
            x + 1.75,
            y + 0.98,
            title,
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color=tone,
        )
        ax.text(
            x + 1.75,
            y + 0.45,
            body,
            ha="center",
            va="center",
            fontsize=11.5,
            color=tone,
            linespacing=1.5,
        )
    for start, end in [
        ((3.72, 3.0), (4.12, 3.0)),
        ((3.72, 0.80), (4.12, 0.80)),
        ((6.0, 1.55), (6.0, 2.23)),
        ((7.87, 3.0), (8.27, 3.0)),
        ((10.15, 2.23), (10.15, 1.55)),
    ]:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "color": COLORS["slate_body"], "lw": 1.7},
        )
    save_figure(fig, output)


def plot_class_size_heatmap(runs: dict[str, Path], output: Path) -> None:
    """Show supported zeros separately from absent GT across the complete vocabulary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    from src.utils.plot_style import add_header, save_figure, set_publication_style

    set_publication_style()
    labels = ["verytiny", "tiny", "small", "medium"]
    datasets = [
        json.loads((path / "evaluation/metrics.json").read_text(encoding="utf-8"))
        for path in runs.values()
    ]
    maximum = max(
        item["AP"] * 100
        for data in datasets
        for c in data["per_class"].values()
        for item in c["bins"].values()
        if item["AP"] is not None
    )
    fig, axes = plt.subplots(1, len(runs), figsize=(12, 7), squeeze=False)
    fig.subplots_adjust(top=0.83, bottom=0.14, right=0.90, wspace=0.36)
    add_header(
        fig,
        "Complete-vocabulary class-by-size output audit",
        "SMOKE ONLY | AP@[.50:.95] in percent; n=eligible GT | Gray=N/A; supported zeros remain zero",
    )
    colorbar = fig.add_axes([0.93, 0.23, 0.015, 0.48])
    cmap = LinearSegmentedColormap.from_list("audit_blues", ["#B9DDF4", "#438BC4", "#123C78"])
    for index, (ax, (name, _), data) in enumerate(zip(axes.flat, runs.items(), datasets)):
        classes = list(data["per_class"].values())
        matrix = np.array(
            [
                [np.nan if c["bins"][a]["AP"] is None else c["bins"][a]["AP"] * 100 for a in labels]
                for c in classes
            ]
        )
        text = np.array(
            [
                [
                    ""
                    if c["bins"][a]["AP"] is None
                    else f"{c['bins'][a]['AP'] * 100:.2f}\nn={c['bins'][a]['support']}"
                    for a in labels
                ]
                for c in classes
            ]
        )
        sns.heatmap(
            matrix,
            ax=ax,
            mask=np.isnan(matrix),
            annot=text,
            fmt="",
            cmap=cmap,
            vmin=0,
            vmax=maximum,
            cbar=index == 0,
            cbar_ax=colorbar if index == 0 else None,
            cbar_kws={"label": "AP (%)"},
            xticklabels=["vt", "t", "s", "m"],
            yticklabels=[c["name"] for c in classes],
            annot_kws={"fontsize": 7.5},
            linewidths=1.2,
            linecolor="#FFFFFF",
        )
        for row, col in zip(*np.where(np.isnan(matrix))):
            ax.add_patch(Rectangle((col, row), 1, 1, facecolor="#E2E8F0", edgecolor="white"))
            ax.text(col + 0.5, row + 0.5, "N/A\nn=0", ha="center", va="center", fontsize=7.5)
        ax.set(title=name)
        ax.set_xlabel("Assignment size bin", fontweight="bold", fontstyle="italic")
        ax.set_ylabel("Category", fontweight="bold", fontstyle="italic")
        ax.tick_params(axis="y", labelrotation=0)
    save_figure(fig, output)


def plot_class_coverage(runs: dict[str, Path], output: Path) -> None:
    """Compare GT coverage with raw output counts without implying TP/recall equivalence."""
    import matplotlib

    matplotlib.use("Agg")
    from collections import Counter

    import matplotlib.pyplot as plt
    import numpy as np

    from src.utils.plot_style import (
        COLORS,
        add_header,
        clean_spines,
        save_figure,
        set_publication_style,
    )

    set_publication_style()
    first = json.loads((next(iter(runs.values())) / "evaluation/metrics.json").read_text())
    ids = list(first["per_class"])
    labels = [first["per_class"][cid]["name"] for cid in ids]
    x = np.arange(len(ids))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.24, wspace=0.23)
    add_header(
        fig,
        "Ground-truth coverage versus smoke output vocabulary",
        "Raw output counts are not true positives | All eight classes remain in the evaluator",
        x=0.055,
    )
    counts = [first["per_class"][cid]["bins"]["all"]["support"] for cid in ids]
    bars = axes[0].bar(x, counts, color=COLORS["slate_light"], width=0.66)
    axes[0].bar_label(bars, padding=4, fontsize=9, fontweight="bold")
    axes[0].set(title="Eligible ground truth", ylabel="GT annotations")
    for index, ((name, path), color) in enumerate(
        zip(runs.items(), [COLORS["indigo"], COLORS["teal"]])
    ):
        record = json.loads((path / "run.json").read_text())
        rows = json.loads(Path(record["predictions"]).read_text())
        raw = Counter(row["category_id"] for row in rows)
        values = [raw[int(cid)] for cid in ids]
        offset = -0.19 if index == 0 else 0.19
        bars = axes[1].bar(x + offset, values, width=0.36, label=name, color=color)
        axes[1].bar_label(
            bars,
            labels=[str(v) if v else "" for v in values],
            padding=4,
            fontsize=8,
            color=color,
            fontweight="bold",
        )
    for i, cid in enumerate(ids):
        if first["per_class"][cid]["name"] not in {"person", "airplane"}:
            axes[1].text(i, 160, "0", ha="center", fontsize=9, color=COLORS["slate_muted"])
    axes[1].set(title="Retained detections (score >= 0.001)", ylabel="Predicted boxes")
    axes[1].legend(loc="upper left", fontsize=9)
    for ax in axes:
        ax.set_xticks(x, [label.replace("-", "-\n") for label in labels], rotation=35, ha="right")
        ax.set_xlim(-0.6, len(ids) - 0.4)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.17)
        ax.tick_params(axis="x", labelsize=8.5)
        clean_spines(ax)
    save_figure(fig, output)


def plot_precision_recall(runs: dict[str, Path], output: Path, category_id: int = 6) -> None:
    """Plot category-specific IoU=.50 interpolated PR; unsupported bins stay explicit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from src.utils.plot_style import (
        SCALE_BIN_PALETTE,
        add_header,
        clean_spines,
        save_figure,
        set_publication_style,
    )

    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.84, bottom=0.12, hspace=0.34, wspace=0.20)
    add_header(
        fig,
        "Person-only diagnostic precision-recall curves",
        "Pretrained smoke outputs | IoU 0.50 | COCO interpolated precision; not trained baseline results",
    )
    for model_index, (name, path) in enumerate(runs.items()):
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        class_name = metrics["per_class"][str(category_id)]["name"]
        if class_name != "person":
            raise ValueError("This diagnostic preview expects the person category")
        with np.load(path / "curves.npz") as data:
            k = int(np.flatnonzero(data["category_ids"] == category_id)[0])
            t = int(np.flatnonzero(np.isclose(data["iou_thresholds"], 0.5))[0])
            for ax, label, color in zip(
                axes.flat, ["verytiny", "tiny", "small", "medium"], ["vt", "t", "s", "m"]
            ):
                a = list(data["area_labels"]).index(label)
                values = data["precision"][t, :, k, a, -1]
                support = metrics["per_class"][str(category_id)]["bins"][label]["support"]
                ax.set_title(
                    f"{label.replace('verytiny', 'very tiny').capitalize()} | GT support: {support}"
                )
                if (values >= 0).any():
                    ax.step(
                        data["recall_thresholds"],
                        np.where(values >= 0, values, np.nan),
                        where="post",
                        color=SCALE_BIN_PALETTE[color],
                        linestyle=["-", "--", ":", "-."][model_index % 4],
                        label=name,
                    )
                elif name == next(iter(runs)):
                    ax.text(0.5, 0.5, "No eligible GT (N/A)", ha="center", transform=ax.transAxes)
                ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel="Recall", ylabel="Precision")
                clean_spines(ax)
                if ax.lines:
                    ax.legend()
    save_figure(fig, output)


def plot_analysis_comparison(analysis: dict, output: Path, *, gaps: bool = False) -> None:
    """Paired AP or signed reference gaps with support kept beside each size label."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from src.utils.plot_style import (
        COLORS,
        add_header,
        clean_spines,
        save_figure,
        set_publication_style,
    )

    set_publication_style()
    labels = ["verytiny", "tiny", "small", "medium"]
    models = analysis["models"]
    first = next(iter(models.values()))
    scope = (
        "SMOKE DIAGNOSTIC ONLY"
        if analysis["contains_smoke"]
        else "Training provenance review required"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.78, bottom=0.19, wspace=0.24)
    title = (
        f"Size-related AP gaps: signed loss versus {analysis['reference_bin']}"
        if gaps
        else "Detection AP by size: all categories versus one class"
    )
    subtitle = (
        f"{scope} | AP@[.50:.95] | Negative loss = improvement; {first['category_name']} medium n={first['class_bins']['medium']['support']}"
        if gaps
        else f"{scope} | Same {first['image_count']} images | Category composition differs across bins"
    )
    add_header(fig, title, subtitle)
    for panel, ax in enumerate(axes):
        for index, ((name, data), color) in enumerate(
            zip(models.items(), [COLORS["indigo"], COLORS["teal"]])
        ):
            values = [
                data["class_gaps"][a][["gap_pp", "relative_loss_percent"][panel]]
                if gaps
                else data[["bins", "class_bins"][panel]][a]["AP"]
                for a in labels
            ]
            values = [np.nan if v is None else v * (1 if gaps else 100) for v in values]
            bars = ax.bar(
                np.arange(4) + (index - 0.5) * 0.36, values, width=0.34, color=color, label=name
            )
            ax.bar_label(
                bars,
                labels=["N/A" if np.isnan(v) else f"{v:.2f}" for v in values],
                padding=4,
                fontsize=8,
            )
        supports = first["class_bins" if gaps or panel else "bins"]
        ax.set_xticks(
            range(4),
            [f"{a.replace('verytiny', 'very tiny')}\nn={supports[a]['support']}" for a in labels],
        )
        ax.set_ylabel(
            ["AP gap (percentage points)", "Relative AP loss (%)"][panel] if gaps else "AP (%)"
        )
        ax.set_title(
            f"{first['category_name']}: {analysis['reference_bin']} reference"
            if gaps
            else ["All eligible categories", first["category_name"].capitalize()][panel]
        )
        ax.axhline(0, color=COLORS["slate_muted"], linewidth=0.8)
        ax.margins(y=0.25)
        ax.legend(fontsize=8)
        clean_spines(ax)
    save_figure(fig, output)


def plot_failure_examples(analysis: dict, coco: dict, image_root: Path, output: Path) -> None:
    """Original-image context and identical crops for deterministically selected GT misses."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    from src.utils.plot_style import COLORS, add_header, save_figure, set_publication_style

    set_publication_style()
    examples, models = analysis["examples"], analysis["models"]
    images = {row["id"]: row for row in coco["images"]}
    category_id = analysis["category_id"]
    score = next(iter(models.values()))["protocol"]["operating_confidence"]
    predictions = {}
    for name, item in models.items():
        record = json.loads((Path(item["run"]) / "run.json").read_text())
        predictions[name] = json.loads(Path(record["predictions"]).read_text())
    fig, axes = plt.subplots(len(examples), 3, figsize=(12, 3.8 * len(examples)), squeeze=False)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.85, bottom=0.08, hspace=0.38, wspace=0.16)
    scope = "SMOKE ONLY" if analysis["contains_smoke"] else "Illustrative diagnostics"
    add_header(
        fig,
        "Illustrative unmatched targets: original context and enlarged crops",
        f"{scope} | IoU 0.50; score >= {score} | Green=target GT; rose=prediction; amber=crop",
    )
    for row, example in enumerate(examples):
        image = images[example["image_id"]]
        with Image.open(image_root / image["file_name"]) as source:
            pixels = source.convert("RGB")
        x, y, w, h = example["bbox"]
        cx, cy = x + w / 2, y + h / 2
        half = max(40, 2 * max(w, h))
        left, right = max(0, cx - half), min(image["width"], cx + half)
        top, bottom = max(0, cy - half), min(image["height"], cy + half)
        for col, ax in enumerate(axes[row]):
            ax.imshow(pixels)
            ax.add_patch(
                Rectangle((x, y), w, h, fill=False, edgecolor=COLORS["emerald"], linewidth=1.7)
            )
            if col == 0:
                ax.add_patch(
                    Rectangle(
                        (left, top),
                        right - left,
                        bottom - top,
                        fill=False,
                        edgecolor=COLORS["amber"],
                        linewidth=1.5,
                    )
                )
                ax.set_title(
                    f"ID {image['id']}: {example['kind'].replace('_', ' ')}\nGT scale {example['scale_px']:.1f} px",
                    fontsize=10,
                )
            else:
                name = list(models)[col - 1]
                shown = [
                    p
                    for p in predictions[name]
                    if p["image_id"] == image["id"]
                    and p["category_id"] == category_id
                    and p["score"] >= score
                    and p["bbox"][0] < right
                    and p["bbox"][1] < bottom
                    and p["bbox"][0] + p["bbox"][2] > left
                    and p["bbox"][1] + p["bbox"][3] > top
                ]
                for p in sorted(shown, key=lambda p: p["score"]):
                    bx, by, bw, bh = p["bbox"]
                    ax.add_patch(
                        Rectangle(
                            (bx, by), bw, bh, fill=False, edgecolor=COLORS["rose"], linewidth=1.2
                        )
                    )
                    ax.text(
                        max(left, bx),
                        max(top, by),
                        f"{p['score']:.2f}",
                        fontsize=7,
                        color=COLORS["rose"],
                        bbox={"facecolor": "white", "alpha": 0.8, "pad": 0.5},
                        clip_on=True,
                    )
                detail = example["by_model"][name]
                status = (
                    "matched at cutoff"
                    if detail is None
                    else f"unmatched; best IoU {detail['best_same_class_iou']:.3f}"
                )
                ax.set_title(f"{name}\n{status}", fontsize=10)
                ax.set_xlim(left, right)
                ax.set_ylim(bottom, top)
            ax.grid(False)
            ax.set_xlabel("Original x (px)", fontsize=8)
            ax.set_ylabel("Original y (px)", fontsize=8)
            ax.tick_params(labelsize=7)
    save_figure(fig, output)
