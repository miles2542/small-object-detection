"""Unit and integration tests for standard high-resolution inference pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image
from ultralytics.engine.results import Results

from src.infer.high_res import (
    HighResConfig,
    HighResInferenceRunner,
    benchmark_resolutions,
    export_high_res_predictions,
    get_peak_memory_mb,
)
from src.infer.utils import CANONICAL_CLASSES, CANONICAL_NAME_TO_ID


def make_results(
    boxes: list[list[float]] | None = None,
    orig_shape: tuple[int, int] = (800, 800),
) -> Results:
    """Create a mock Ultralytics Results object."""
    values = [[100.0, 150.0, 140.0, 190.0, 0.85, 0.0]] if boxes is None else boxes
    res = Results(
        orig_img=np.zeros((*orig_shape, 3), dtype=np.uint8),
        path="sample.png",
        names=CANONICAL_CLASSES,
        boxes=torch.tensor(values, dtype=torch.float32).reshape(-1, 6) if values else torch.zeros((0, 6)),
    )
    res.speed = {"preprocess": 1.5, "inference": 12.0, "postprocess": 2.0}
    return res


class TestHighResConfig(unittest.TestCase):
    """Test validation and boundary checks in HighResConfig."""

    def test_default_config_valid(self) -> None:
        cfg = HighResConfig()
        self.assertEqual(cfg.image_size, 800)
        self.assertEqual(cfg.confidence_threshold, 0.001)
        self.assertEqual(cfg.iou_threshold, 0.70)
        self.assertEqual(cfg.max_detections_per_image, 1500)
        self.assertEqual(cfg.batch_size, 1)

    def test_invalid_image_size(self) -> None:
        with self.assertRaises(ValueError):
            HighResConfig(image_size=0)
        with self.assertRaises(ValueError):
            HighResConfig(image_size=-640)

    def test_invalid_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            HighResConfig(confidence_threshold=1.5)
        with self.assertRaises(ValueError):
            HighResConfig(iou_threshold=-0.1)

    def test_invalid_limits(self) -> None:
        with self.assertRaises(ValueError):
            HighResConfig(max_detections_per_image=0)
        with self.assertRaises(ValueError):
            HighResConfig(batch_size=0)


class TestMemoryMonitoring(unittest.TestCase):
    """Test memory reporting function."""

    def test_get_peak_memory_mb(self) -> None:
        mem = get_peak_memory_mb("cpu")
        self.assertIsInstance(mem, float)
        self.assertGreater(mem, 0.0)


class TestHighResInferenceRunner(unittest.TestCase):
    """Test prediction formatting, box conversion, sorting, and capping."""

    def setUp(self) -> None:
        self.mock_model = MagicMock()
        self.mock_model.names = CANONICAL_CLASSES

    def test_predict_image_formatting(self) -> None:
        # 2 boxes: [x1, y1, x2, y2, score, class]
        boxes = [
            [50.0, 60.0, 80.0, 100.0, 0.75, 0.0],  # airplane: w=30, h=40
            [200.0, 300.0, 220.0, 350.0, 0.95, 6.0],  # person: w=20, h=50
        ]
        self.mock_model.predict.return_value = [make_results(boxes)]

        runner = HighResInferenceRunner(model=self.mock_model)
        img = np.zeros((800, 800, 3), dtype=np.uint8)
        detections, durations = runner.predict_image(img, image_id=10)

        self.assertEqual(len(detections), 2)
        # Should be sorted descending by score (person first with 0.95)
        self.assertEqual(detections[0]["image_id"], 10)
        self.assertEqual(detections[0]["category_id"], CANONICAL_NAME_TO_ID["person"])
        self.assertEqual(detections[0]["bbox"], [200.0, 300.0, 20.0, 50.0])
        self.assertAlmostEqual(detections[0]["score"], 0.95)

        self.assertEqual(detections[1]["category_id"], CANONICAL_NAME_TO_ID["airplane"])
        self.assertEqual(detections[1]["bbox"], [50.0, 60.0, 30.0, 40.0])
        self.assertAlmostEqual(detections[1]["score"], 0.75)

        self.assertIn("preprocess_ms", durations)
        self.assertIn("inference_ms", durations)
        self.assertIn("postprocess_ms", durations)
        self.assertIn("total_ms", durations)
        self.assertIn("peak_memory_mb", durations)

    def test_empty_detections(self) -> None:
        self.mock_model.predict.return_value = [make_results([])]

        runner = HighResInferenceRunner(model=self.mock_model)
        img = np.zeros((800, 800, 3), dtype=np.uint8)
        detections, durations = runner.predict_image(img, image_id=20)

        self.assertEqual(detections, [])
        self.assertGreater(durations["total_ms"], 0.0)

    def test_detection_capping(self) -> None:
        boxes = [[10.0, 10.0, 20.0, 20.0, float(i) / 10.0, 0.0] for i in range(1, 11)]
        self.mock_model.predict.return_value = [make_results(boxes)]

        cfg = HighResConfig(max_detections_per_image=3)
        runner = HighResInferenceRunner(model=self.mock_model, config=cfg)
        detections, _ = runner.predict_image(np.zeros((800, 800, 3), dtype=np.uint8), image_id=30)

        self.assertEqual(len(detections), 3)
        self.assertAlmostEqual(detections[0]["score"], 1.0)
        self.assertAlmostEqual(detections[1]["score"], 0.9)
        self.assertAlmostEqual(detections[2]["score"], 0.8)


class TestExportHighResPredictions(unittest.TestCase):
    """Test full dataset split export and resolution benchmark."""

    def test_export_and_benchmark_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            weights_file = tmp_path / "model.pt"
            weights_file.write_bytes(b"dummy_weights")

            img_dir = tmp_path / "images"
            img_dir.mkdir()
            img_file = img_dir / "aerial.png"
            Image.fromarray(np.zeros((800, 800, 3), dtype=np.uint8)).save(img_file)

            ann_file = tmp_path / "ann.json"
            ann_data = {
                "images": [{"id": 501, "file_name": "aerial.png", "width": 800, "height": 800}],
                "categories": [{"id": cid, "name": name} for cid, name in CANONICAL_CLASSES.items()],
                "annotations": [],
            }
            ann_file.write_text(json.dumps(ann_data), encoding="utf-8")

            yaml_file = tmp_path / "dataset.yaml"
            yaml_content = (
                f"path: {tmp_path.as_posix()}\n"
                f"val: images\n"
                f"names:\n" + "\n".join(f"  {k}: {v}" for k, v in CANONICAL_CLASSES.items())
            )
            yaml_file.write_text(yaml_content, encoding="utf-8")

            output_dir = tmp_path / "output_high_res"

            with patch("src.infer.high_res.HighResInferenceRunner") as MockRunnerClass:
                mock_runner = MagicMock()
                mock_runner.predict_image.return_value = (
                    [
                        {
                            "image_id": 501,
                            "category_id": 3,
                            "bbox": [100.0, 120.0, 45.0, 18.0],
                            "score": 0.82,
                        }
                    ],
                    {
                        "preprocess_ms": 1.0,
                        "inference_ms": 15.0,
                        "postprocess_ms": 2.0,
                        "total_ms": 18.0,
                        "peak_memory_mb": 150.0,
                    },
                )
                MockRunnerClass.return_value = mock_runner

                # Test export at single resolution
                manifest = export_high_res_predictions(
                    weights=weights_file,
                    data=yaml_file,
                    output_dir=output_dir,
                    image_size=1280,
                    split="val",
                    annotations=ann_file,
                    image_root=img_dir,
                )

                self.assertTrue((output_dir / "predictions.json").is_file())
                self.assertTrue((output_dir / "manifest.json").is_file())
                self.assertEqual(manifest["image_size"], 1280)
                self.assertEqual(manifest["strategy"], "high_res_standard")
                self.assertIn("throughput_fps", manifest["latency"])
                self.assertIn("peak_mb", manifest["memory"])

                # Test resolution benchmark sweep
                bench_dir = tmp_path / "output_bench"
                summary = benchmark_resolutions(
                    weights=weights_file,
                    data=yaml_file,
                    output_dir=bench_dir,
                    resolutions=[800, 1280],
                    split="val",
                    annotations=ann_file,
                    image_root=img_dir,
                )

                self.assertTrue((bench_dir / "resolution_benchmark.json").is_file())
                self.assertIn("800", summary["resolutions"])
                self.assertIn("1280", summary["resolutions"])


if __name__ == "__main__":
    unittest.main()
