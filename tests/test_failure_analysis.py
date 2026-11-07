"""End-to-end TIDE attribution and saved-prediction CLI checks."""

import json
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from src.cli import app
from src.eval.audit import sha256
from src.eval.failures import analyze_failures, exclusive_size_bin


@pytest.fixture
def sample(tmp_path: Path) -> tuple[Path, Path, Path]:
    Image.new("RGB", (120, 120), "white").save(tmp_path / "frame.png")
    ground_truth = {
        "images": [{"id": 1, "file_name": "frame.png", "width": 120, "height": 120}],
        "categories": [{"id": 0, "name": "object"}],
        "annotations": [
            {
                "id": 0,
                "image_id": 1,
                "category_id": 0,
                "bbox": [10, 10, 4, 4],
                "area": 16,
                "iscrowd": 0,
            },
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [50, 50, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
        ],
    }
    predictions = [
        {"image_id": 1, "category_id": 0, "bbox": [12, 10, 4, 4], "score": 0.9},
        {"image_id": 1, "category_id": 0, "bbox": [90, 90, 5, 5], "score": 0.8},
    ]
    gt_path, pred_path = tmp_path / "gt.json", tmp_path / "predictions.json"
    gt_path.write_text(json.dumps(ground_truth), encoding="utf-8")
    pred_path.write_text(json.dumps(predictions), encoding="utf-8")
    return gt_path, pred_path, tmp_path


def test_tide_error_attribution_and_gallery(sample: tuple[Path, Path, Path]) -> None:
    gt_path, pred_path, root = sample
    result = analyze_failures(gt_path, {"baseline": pred_path}, root / "analysis", image_root=root)
    counts = result["models"]["baseline"]["error_counts"]
    assert counts == {"Cls": 0, "Loc": 1, "Both": 0, "Dupe": 0, "Bkg": 1, "Miss": 1}
    assert result["models"]["baseline"]["size_bins"]["verytiny"]["gt_support"] == 1
    assert result["models"]["baseline"]["size_bins"]["tiny"]["gt_support"] == 1
    assert result["models"]["baseline"]["size_bins"]["verytiny"]["loc_center_shift_median_px"] == 2
    assert {item["type"] for item in result["gallery_examples"]} == {"Loc", "Miss", "Bkg"}
    for name in (
        "failure_analysis.json",
        "report.md",
        "tide_breakdown.png",
        "failure_by_size.png",
        "failure_gallery.png",
        "gallery_examples.json",
    ):
        assert (root / "analysis" / name).is_file()
    assert "Official COCO AP" in (root / "analysis" / "report.md").read_text()


def test_exclusive_event_bins_and_manifest_guard(sample: tuple[Path, Path, Path]) -> None:
    gt_path, pred_path, root = sample
    assert [exclusive_size_bin(area) for area in (63.99, 64, 256, 1024)] == [
        "verytiny",
        "tiny",
        "small",
        "medium",
    ]
    rows = json.loads(pred_path.read_text())
    manifest = {
        "status": "complete",
        "predictions_sha256": sha256(pred_path),
        "annotations": {"sha256": sha256(gt_path)},
        "detections": len(rows),
        "image_ids": [1],
        "prediction_counts": {"1": len(rows)},
        "predict_settings": {"imgsz": 800, "conf": 0.001, "max_det": 1500, "augment": False},
        "category_mapping": {"0": {"category_id": 0}},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = analyze_failures(gt_path, {"baseline": pred_path}, root / "verified")
    assert result["models"]["baseline"]["verified_manifest"]
    manifest["prediction_counts"]["1"] = 0
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="counts disagree"):
        analyze_failures(gt_path, {"baseline": pred_path}, root / "bad")


def test_cli_saved_prediction_run(sample: tuple[Path, Path, Path]) -> None:
    gt_path, pred_path, root = sample
    outcome = CliRunner().invoke(
        app,
        [
            "eval",
            "failures",
            "--ground-truth",
            str(gt_path),
            "--model",
            f"baseline={pred_path}",
            "--output-dir",
            str(root / "cli"),
        ],
    )
    assert outcome.exit_code == 0, outcome.output
    assert (root / "cli" / "failure_analysis.json").is_file()


def test_empty_predictions_remain_a_valid_failure_run(sample: tuple[Path, Path, Path]) -> None:
    gt_path, pred_path, root = sample
    pred_path.write_text("[]", encoding="utf-8")
    result = analyze_failures(gt_path, {"empty": pred_path}, root / "empty")
    assert result["models"]["empty"]["error_counts"]["Miss"] == 2
    assert result["models"]["empty"]["tide_ap50_percent"] == 0


def test_all_six_tide_modes_are_exported(tmp_path: Path) -> None:
    gt_boxes = [[0, 0, 10, 10], [30, 0, 10, 10], [60, 0, 10, 10], [90, 0, 10, 10], [0, 30, 10, 10]]
    coco = {
        "images": [{"id": 1, "file_name": "frame.png", "width": 120, "height": 120}],
        "categories": [{"id": 0, "name": "target"}, {"id": 1, "name": "other"}],
        "annotations": [
            {"id": i, "image_id": 1, "category_id": 0, "bbox": box, "area": 100, "iscrowd": 0}
            for i, box in enumerate(gt_boxes)
        ],
    }
    detections = [
        (0, 0.99, [0, 0, 10, 10]),  # true positive
        (0, 0.98, [0, 0, 10, 10]),  # Dupe
        (1, 0.90, [30, 0, 10, 10]),  # Cls
        (0, 0.80, [64, 0, 10, 10]),  # Loc
        (1, 0.70, [94, 0, 10, 10]),  # Both
        (0, 0.60, [60, 60, 10, 10]),  # Bkg
    ]
    rows = [
        {"image_id": 1, "category_id": category, "score": score, "bbox": box}
        for category, score, box in detections
    ]
    gt_path, pred_path = tmp_path / "gt.json", tmp_path / "pred.json"
    gt_path.write_text(json.dumps(coco), encoding="utf-8")
    pred_path.write_text(json.dumps(rows), encoding="utf-8")
    result = analyze_failures(gt_path, {"model": pred_path}, tmp_path / "out")
    assert result["models"]["model"]["error_counts"] == {
        "Cls": 1,
        "Loc": 1,
        "Both": 1,
        "Dupe": 1,
        "Bkg": 1,
        "Miss": 2,
    }
