"""Small orchestration layer shared by the evaluation CLI and Python callers."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.eval.audit import sha256
from src.eval.coco_eval import evaluate_files, validate_inputs
from src.eval.protocol import evaluation_protocol, prepare_ground_truth


def resolve_inputs(
    data: Path, split: str, annotations: Path | None, image_root: Path | None
) -> tuple[Path, Path | None, list[int] | None]:
    """YAML selects actual image files; explicit COCO JSON selects its image records."""
    if data.suffix.lower() == ".json":
        if annotations is not None:
            raise ValueError("--data is already COCO JSON; omit --annotations")
        return data, image_root, None
    if data.suffix.lower() not in [".yaml", ".yml"]:
        raise ValueError("--data must be dataset YAML or COCO annotation JSON")
    try:
        config = yaml.safe_load(data.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid dataset YAML: {error}") from error
    if not isinstance(config, dict) or not isinstance(config.get(split), str):
        raise TypeError(f"Dataset YAML needs a single directory entry for split {split!r}")
    if image_root is None:
        base = Path(config.get("path", data.parent.resolve()))
        if not base.is_absolute():
            candidates = {(data.parent / base).resolve(), base.resolve()}
            candidates = {path for path in candidates if path.is_dir()}
            if len(candidates) != 1:
                raise ValueError("Dataset root is missing or ambiguous; use an absolute YAML path")
            base = candidates.pop()
        image_root = base / config[split]
    image_root = image_root.resolve()
    if not image_root.is_dir():
        raise ValueError(f"Image directory not found: {image_root}")
    if annotations is None:
        names = {"train": "aitodv2_train.json", "val": "aitod_val.json"}
        filename = names.get(split)
        annotations = next(
            (
                parent / "annotations" / filename
                for parent in image_root.parents
                if filename and (parent / "annotations" / filename).is_file()
            ),
            None,
        )
        if annotations is None:
            raise ValueError("Cannot infer COCO annotations; provide --annotations explicitly")
    coco = json.loads(annotations.read_text(encoding="utf-8"))
    configured = config.get("names", {})
    configured = list(configured.values()) if isinstance(configured, dict) else configured
    if len(configured) != len(coco["categories"]) or set(configured) != {
        category["name"] for category in coco["categories"]
    }:
        raise ValueError("Dataset YAML and COCO category names disagree")
    files = {
        path.relative_to(image_root).as_posix()
        for path in image_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    }
    names = [image["file_name"] for image in coco["images"]]
    if len(set(names)) != len(names) or not files or not files <= set(names):
        raise ValueError("Image directory has no images, duplicate COCO filenames or unknown files")
    selected = [image["id"] for image in coco["images"] if image["file_name"] in files]
    return annotations, image_root, selected


def run_evaluation(
    data: Path,
    output_dir: Path,
    *,
    weights: Path | None = None,
    predictions: Path | None = None,
    split: str = "val",
    annotations: Path | None = None,
    image_root: Path | None = None,
    protocol_name: str = "assignment",
    smoke: bool = False,
    device: str = "cpu",
    batch_size: int = 4,
    confidence: float | None = None,
) -> dict:
    """Evaluate saved JSON or export a checkpoint first; never substitute weights."""
    data, output_dir = Path(data), Path(output_dir)
    if (weights is None) == (predictions is None):
        raise ValueError("Choose exactly one of --weights and --predictions")
    if split not in ["train", "val", "test"]:
        raise ValueError("--split must be train, val or test")
    protocol = evaluation_protocol(protocol_name)
    if (
        weights is not None
        and confidence is not None
        and confidence < protocol["export_confidence"]
    ):
        raise ValueError("Operating confidence is below the prediction export floor")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    if predictions is not None and (data.suffix.lower() != ".json" or annotations or image_root):
        raise ValueError(
            "Saved predictions require --data COCO_JSON; image/annotation overrides are unnecessary"
        )
    source, images, selected = resolve_inputs(data, split, annotations, image_root)
    if weights is not None and images is None:
        raise ValueError("COCO JSON with --weights requires --image-root")
    source_hash = sha256(source)
    if weights is not None:
        from src.eval.export import export_predictions

        coco = json.loads(source.read_text(encoding="utf-8"))
        validate_inputs(coco, [], selected)
        prepared = prepare_ground_truth(coco, selected)
        output_dir.mkdir(parents=True, exist_ok=True)
        source = output_dir / "ground_truth.json"
        source.write_text(json.dumps(prepared), encoding="utf-8")
        export_predictions(
            weights,
            source,
            images,
            output_dir / "prediction_export",
            smoke=smoke,
            protocol_name=protocol_name,
            device=device,
            batch_size=batch_size,
        )
        predictions = output_dir / "prediction_export/predictions.json"
    result = evaluate_files(
        source,
        predictions,
        output_dir / "evaluation",
        protocol_name=protocol_name,
        allow_smoke=smoke,
        confidence=confidence,
    )
    record = {
        "status": "complete",
        "mode": "weights" if weights else "saved_predictions",
        "data": data.as_posix(),
        "data_sha256": sha256(data),
        "split_label": split,
        "source_annotations_sha256": source_hash,
        "evaluated_annotations": source.as_posix(),
        "predictions": Path(predictions).as_posix(),
        "image_count": len(result["image_ids"]),
        "purpose": result["provenance"]["purpose"],
        "protocol": result["protocol"],
        "runner_code_sha256": sha256(Path(__file__)),
    }
    (output_dir / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return result
