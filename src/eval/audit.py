"""Audit prepared AI-TOD inputs without inference: python -m src.eval.audit."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

from PIL import Image

from src.data.subset import AITOD_CLASSES, get_scale_bin, select_stratified_subset
from src.eval.protocol import (
    REFERENCE_SHA256,
    REFERENCE_URL,
    evaluation_protocol,
    prepare_ground_truth,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(annotations: list[dict]) -> dict:
    valid = [a for a in annotations if a["bbox"][2] > 0 and a["bbox"][3] > 0 and a["area"] > 0]
    return {
        "valid_boxes": len(valid),
        "classes": dict(Counter(str(a["category_id"]) for a in valid)),
        "scales": dict(
            Counter(get_scale_bin(math.sqrt(a["bbox"][2] * a["bbox"][3])) for a in valid)
        ),
    }


def percentage_drift(full: dict, mini: dict, field: str) -> dict:
    """Mini minus full object-frequency shares, in percentage points."""
    return {
        key: 100
        * (
            mini[field].get(key, 0) / mini["valid_boxes"]
            - full[field].get(key, 0) / full["valid_boxes"]
        )
        for key in full[field]
    }


def audit_split(root: Path, split: str, coco: dict, selected: list[int]) -> tuple[dict, dict]:
    """Check IDs, geometry, split coverage and actual image content in one pass."""
    images, annotations = coco["images"], coco["annotations"]
    ids = {image["id"] for image in images}
    names = [image["file_name"] for image in images]
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    if (
        len(ids) != len(images)
        or len(set(names)) != len(names)
        or len({Path(name).stem for name in names}) != len(names)
        or len({a["id"] for a in annotations}) != len(annotations)
    ):
        raise ValueError(f"{split}: duplicate IDs, filenames or filename stems")
    if set(categories.values()) != set(AITOD_CLASSES) or len(categories) != 8:
        raise ValueError(f"{split}: unexpected category vocabulary")
    if len(set(selected)) != len(selected) or not set(selected) <= ids:
        raise ValueError(f"{split}: invalid selected image IDs")
    invalid, outside, area_mismatch, subpixel = [], [], [], []
    by_image = {image["id"]: image for image in images}
    for ann in annotations:
        box = ann["bbox"]
        if (
            ann["image_id"] not in ids
            or ann["category_id"] not in categories
            or len(box) != 4
            or not all(math.isfinite(x) for x in box + [ann["area"]])
        ):
            raise ValueError(f"{split}: malformed annotation {ann['id']}")
        x, y, w, h = box
        if w <= 0 or h <= 0 or ann["area"] <= 0:
            invalid.append(ann["id"])
            continue
        if not math.isclose(ann["area"], w * h, rel_tol=1e-9, abs_tol=1e-6):
            area_mismatch.append(ann["id"])
        if min(w, h) < 1:
            subpixel.append(ann["id"])
        image = by_image[ann["image_id"]]
        if x < 0 or y < 0 or x + w > image["width"] or y + h > image["height"]:
            outside.append(ann["id"])
    if area_mismatch:
        raise ValueError(f"{split}: annotated area differs from bbox area")
    raw = root / "images" / split
    if {p.name for p in raw.glob("*.png")} != set(names):
        raise ValueError(f"{split}: raw image coverage differs from annotations")

    def inspect(image: dict) -> tuple[str, str]:
        path = raw / image["file_name"]
        with Image.open(path) as decoded:
            if decoded.size != (image["width"], image["height"]):
                raise ValueError(f"Image dimension mismatch: {path}")
        return image["file_name"], sha256(path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(inspect, images))
    selected_set = set(selected)
    mini_annotations = [a for a in annotations if a["image_id"] in selected_set]
    full, mini = distribution(annotations), distribution(mini_annotations)
    selected_names = {image["file_name"] for image in images if image["id"] in selected_set}
    for tag, expected in [("full", set(names)), ("mini", selected_names)]:
        prepared = root / "splits" / tag
        if {p.name for p in (prepared / "images" / split).glob("*.png")} != expected:
            raise ValueError(f"{tag}/{split}: image selection mismatch")
        if {p.stem for p in (prepared / "labels" / split).glob("*.txt")} != {
            Path(name).stem for name in expected
        }:
            raise ValueError(f"{tag}/{split}: label coverage mismatch")
    # Training images are hard links; validate copied development validation images too.
    if split == "val":
        with ThreadPoolExecutor(max_workers=4) as pool:
            copied = dict(
                zip(
                    sorted(selected_names),
                    pool.map(
                        sha256,
                        [root / "splits/mini/images/val" / n for n in sorted(selected_names)],
                    ),
                )
            )
        if any(copied[name] != hashes[name] for name in copied):
            raise ValueError("Mini validation image content differs from source")
    regenerated, _ = select_stratified_subset(coco, len(selected), seed=42)
    if regenerated != selected:
        raise ValueError(f"{split}: seed-42 subset is not reproducible")
    return {
        "images": len(images),
        "annotations": len(annotations),
        "mini_images": len(selected),
        "invalid_annotation_ids": invalid,
        "outside_image_annotation_ids": outside,
        "subpixel_positive_boxes": len(subpixel),
        "area_mismatches": len(area_mismatch),
        "annotation_id_zero": any(a["id"] == 0 for a in annotations),
        "crowd_counts": dict(Counter(str(a.get("iscrowd", 0)) for a in annotations)),
        "ignore_counts": dict(Counter(str(a.get("ignore", 0)) for a in annotations)),
        "max_objects_per_image": max(Counter(a["image_id"] for a in annotations).values()),
        "class_index_to_category_id": {
            str(i): next(cid for cid, name in categories.items() if name == label)
            for i, label in enumerate(AITOD_CLASSES)
        },
        "full": full,
        "mini": mini,
        "class_drift_pp": percentage_drift(full, mini, "classes"),
        "scale_drift_pp": percentage_drift(full, mini, "scales"),
    }, hashes


def plot_audit(summary: dict, output: Path = Path("runs/eval/input_audit/validation_subset_audit.png")) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")  # Save-only reporting must not depend on a GUI/Tk installation.
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(top=0.78, bottom=0.26, wspace=0.30)
    add_header(
        fig,
        "Validation subset representativeness",
        "AI-TOD-v2 | 200-image development subset versus 2,804-image validation set",
    )
    audit = summary["splits"]["val"]
    for ax, field, labels in zip(
        axes, ["classes", "scales"], [AITOD_CLASSES, ["vt", "t", "s", "m", "l"]]
    ):
        keys = (
            [str(audit["class_index_to_category_id"][str(i)]) for i in range(8)]
            if field == "classes"
            else labels
        )
        x = np.arange(len(labels))
        for offset, tag, color in [
            (-0.19, "full", COLORS["indigo"]),
            (0.19, "mini", COLORS["teal"]),
        ]:
            values = [100 * audit[tag][field].get(k, 0) / audit[tag]["valid_boxes"] for k in keys]
            ax.bar(x + offset, values, 0.38, label=tag.capitalize(), color=color)
        ax.set_xticks(x, labels, rotation=45 if field == "classes" else 0, ha="right")
        ax.set_ylabel("Share of valid boxes (%)")
        ax.set_title("Category composition" if field == "classes" else "Exclusive EDA scale bins")
        ax.legend()
        clean_spines(ax)
    save_figure(fig, output)


def main() -> None:
    root, output = Path("data/ai-tod-v2"), Path("runs/eval/input_audit")
    output.mkdir(parents=True, exist_ok=True)
    preparation = json.loads((root / "preparation_summary.json").read_text())
    summary = {
        "seed": 42,
        "execution_device": "CPU; no model inference",
        "python": platform.python_version(),
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "versions": {p: version(p) for p in ["torch", "ultralytics", "numpy", "Pillow"]},
        "protocol": evaluation_protocol(),
        "reference": {"url": REFERENCE_URL, "sha256": REFERENCE_SHA256},
        "code_sha256": {
            name: sha256(Path(name)) for name in ["src/eval/audit.py", "src/eval/protocol.py"]
        },
        "splits": {},
    }
    hashes, filenames = {}, {}
    for split, filename in [("train", "aitodv2_train.json"), ("val", "aitod_val.json")]:
        path = root / "annotations" / filename
        coco = json.loads(path.read_text())
        result, hashes[split] = audit_split(
            root, split, coco, preparation[f"mini_{split}_image_ids"]
        )
        result["annotation_sha256"] = sha256(path)
        if result["annotation_sha256"] != preparation["annotation_sha256"][split]:
            raise ValueError(f"{split}: annotation hash changed since preparation")
        summary["splits"][split] = result
        filenames[split] = set(hashes[split])
        if split == "val":
            for tag, selection in [("val", None), ("mini_val", preparation["mini_val_image_ids"])]:
                ground_truth = prepare_ground_truth(coco, selection)
                target = output / f"{tag}_ground_truth.json"
                target.write_text(json.dumps(ground_truth), encoding="utf-8")
        print(
            f"PASS {split}: {result['images']} images, {result['annotations']} annotations",
            flush=True,
        )
    summary["filename_overlap"] = len(filenames["train"] & filenames["val"])
    summary["content_hash_overlap"] = len(
        set(hashes["train"].values()) & set(hashes["val"].values())
    )
    if summary["filename_overlap"] or summary["content_hash_overlap"]:
        raise ValueError("Train/validation leakage detected; inspect image hashes")
    summary["checkpoints"] = [
        {"path": p.as_posix(), "sha256": sha256(p)}
        for base in [Path("runs/train"), Path("runs/kaggle"), Path("weights")]
        for p in base.rglob("*.pt")
    ]
    (output / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "image_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    plot_audit(summary, output / "validation_subset_audit.png")
    print(f"PASS: input audit saved to {output}; train/validation sets are disjoint", flush=True)


if __name__ == "__main__":
    main()
