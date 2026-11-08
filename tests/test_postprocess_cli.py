"""Integration tests for sod postprocess CLI subcommands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

from src.cli import app


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


def test_cli_postprocess_help(cli_runner: CliRunner) -> None:
    """Verifies sod postprocess --help lists fuse, tta, and bench commands."""
    result = cli_runner.invoke(app, ["postprocess", "--help"])
    assert result.exit_code == 0
    assert "fuse" in result.output
    assert "tta" in result.output
    assert "bench" in result.output


def test_cli_postprocess_fuse(tmp_path: Path, cli_runner: CliRunner) -> None:
    """Verifies sod postprocess fuse combines prediction JSONs into valid COCO format."""
    pred1_path = tmp_path / "model1_preds.json"
    pred2_path = tmp_path / "model2_preds.json"
    output_path = tmp_path / "fused.json"

    # Model 1 detects vehicle (category 5) at [10, 10, 20, 20]
    preds1 = [
        {"image_id": 1, "category_id": 5, "bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.85}
    ]
    # Model 2 detects identical vehicle at [11, 10, 20, 20]
    preds2 = [
        {"image_id": 1, "category_id": 5, "bbox": [11.0, 10.0, 20.0, 20.0], "score": 0.90}
    ]

    pred1_path.write_text(json.dumps(preds1), encoding="utf-8")
    pred2_path.write_text(json.dumps(preds2), encoding="utf-8")

    result = cli_runner.invoke(
        app,
        [
            "postprocess",
            "fuse",
            "-i",
            str(pred1_path),
            "-i",
            str(pred2_path),
            "-w",
            "1.0",
            "-w",
            "1.0",
            "--iou-threshold",
            "0.50",
            "--conf-threshold",
            "0.20",
            "-o",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.is_file()

    fused_data = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(fused_data, list)
    assert len(fused_data) == 1
    det = fused_data[0]
    assert det["image_id"] == 1
    assert det["category_id"] == 5
    assert len(det["bbox"]) == 4
    # Averaged x should be ~10.5
    assert 10.0 <= det["bbox"][0] <= 11.0
    assert det["score"] >= 0.85


def test_cli_postprocess_bench(tmp_path: Path, cli_runner: CliRunner) -> None:
    """Verifies sod postprocess bench evaluates predictions against COCO ground truth."""
    gt_path = tmp_path / "ground_truth.json"
    pred_path = tmp_path / "predictions.json"
    bench_out = tmp_path / "bench_results"

    gt_data = {
        "images": [{"id": 42, "file_name": "target.png", "width": 800, "height": 800}],
        "categories": [{"id": 0, "name": "airplane"}, {"id": 5, "name": "vehicle"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 42,
                "category_id": 5,
                "bbox": [100.0, 100.0, 15.0, 15.0],
                "area": 225.0,
                "iscrowd": 0,
            }
        ],
    }
    pred_data = [
        {"image_id": 42, "category_id": 5, "bbox": [100.0, 100.0, 15.0, 15.0], "score": 0.95}
    ]

    gt_path.write_text(json.dumps(gt_data), encoding="utf-8")
    pred_path.write_text(json.dumps(pred_data), encoding="utf-8")

    result = cli_runner.invoke(
        app,
        [
            "postprocess",
            "bench",
            "-g",
            str(gt_path),
            "-p",
            str(pred_path),
            "-o",
            str(bench_out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "AI-TOD Micro-Scale Benchmark" in result.output
    assert "Scale-Partitioned Performance" in result.output
    assert (bench_out / "metrics.json").is_file()


def test_cli_postprocess_tta(tmp_path: Path, cli_runner: CliRunner) -> None:
    """Verifies sod postprocess tta executes with lazy weight loading and produces predictions."""
    # Create a dummy image
    img_path = tmp_path / "0000042.png"
    img = Image.new("RGB", (800, 800), color=(128, 128, 128))
    img.save(img_path)

    dummy_weights = tmp_path / "mock_weights.pt"
    dummy_weights.write_bytes(b"mock_binary_weights")
    output_path = tmp_path / "tta_out.json"

    # Mock Ultralytics YOLO model
    mock_boxes = MagicMock()
    mock_boxes.xyxy = torch.tensor([[100.0, 100.0, 120.0, 120.0]])
    mock_boxes.conf = torch.tensor([0.92])
    mock_boxes.cls = torch.tensor([5.0])

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes

    mock_yolo_instance = MagicMock()
    mock_yolo_instance.predict.return_value = [mock_result]
    mock_yolo_instance.names = {5: "vehicle"}

    with patch("ultralytics.YOLO", return_value=mock_yolo_instance):
        result = cli_runner.invoke(
            app,
            [
                "postprocess",
                "tta",
                "-w",
                str(dummy_weights),
                "-s",
                str(img_path),
                "--scales",
                "1.0",
                "--hflip",
                "--no-vflip",
                "-o",
                str(output_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert output_path.is_file()
    tta_data = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(tta_data) >= 1
    assert tta_data[0]["image_id"] == 42


def test_cli_postprocess_error_handling(tmp_path: Path, cli_runner: CliRunner) -> None:
    """Verifies that nonexistent files trigger error exits."""
    missing_file = tmp_path / "missing.json"

    result = cli_runner.invoke(
        app,
        ["postprocess", "fuse", "-i", str(missing_file)],
    )
    assert result.exit_code != 0
