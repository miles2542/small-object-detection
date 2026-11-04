"""Unit and contract tests for SAHI slicing inference engine."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
from sahi.prediction import ObjectPrediction, PredictionResult

from src.infer.sahi import (
    SahiConfig,
    SahiInferenceRunner,
    export_sahi_predictions,
)
from src.infer.utils import (
    CANONICAL_CLASSES,
    CANONICAL_NAME_TO_ID,
    build_category_mapping,
)


def make_object_prediction(
    box: list[float],
    cat_id: int,
    cat_name: str,
    score: float,
) -> ObjectPrediction:
    """Helper to instantiate a valid ObjectPrediction."""
    return ObjectPrediction(
        bbox=box,
        category_id=cat_id,
        category_name=cat_name,
        score=score,
    )


class TestSahiConfig(unittest.TestCase):
    """Test validation and boundary checks in SahiConfig."""

    def test_default_config_valid(self) -> None:
        cfg = SahiConfig()
        self.assertEqual(cfg.slice_height, 512)
        self.assertEqual(cfg.slice_width, 512)
        self.assertEqual(cfg.overlap_height_ratio, 0.20)
        self.assertEqual(cfg.overlap_width_ratio, 0.20)
        self.assertEqual(cfg.confidence_threshold, 0.001)
        self.assertEqual(cfg.max_detections_per_image, 1500)

    def test_invalid_slice_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            SahiConfig(slice_height=0)
        with self.assertRaises(ValueError):
            SahiConfig(slice_width=-100)

    def test_invalid_overlap_ratios(self) -> None:
        with self.assertRaises(ValueError):
            SahiConfig(overlap_height_ratio=-0.1)
        with self.assertRaises(ValueError):
            SahiConfig(overlap_width_ratio=1.0)

    def test_invalid_postprocess_parameters(self) -> None:
        with self.assertRaises(ValueError):
            SahiConfig(postprocess_type="UNKNOWN_NMS")
        with self.assertRaises(ValueError):
            SahiConfig(postprocess_match_metric="INVALID")
        with self.assertRaises(ValueError):
            SahiConfig(confidence_threshold=1.5)


class TestCategoryMapping(unittest.TestCase):
    """Test category mapping between model names and AI-TOD canonical IDs."""

    def test_exact_canonical_mapping(self) -> None:
        mapping = build_category_mapping(CANONICAL_CLASSES)
        for cid in CANONICAL_CLASSES:
            self.assertEqual(mapping[cid], cid)

    def test_reordered_mapping(self) -> None:
        custom_names = {0: "vehicle", 1: "airplane"}
        target = [{"id": 5, "name": "vehicle"}, {"id": 0, "name": "airplane"}]
        mapping = build_category_mapping(custom_names, target_categories=target)
        self.assertEqual(mapping[0], 5)
        self.assertEqual(mapping[1], 0)

    def test_vocabulary_disagreement_without_smoke(self) -> None:
        with self.assertRaises(ValueError):
            build_category_mapping({0: "car", 1: "pedestrian"})

    def test_smoke_partial_vocabulary(self) -> None:
        mapping = build_category_mapping(
            {0: "airplane", 1: "unknown_class"},
            smoke=True,
        )
        self.assertEqual(mapping, {0: 0})


class TestSahiInferenceRunner(unittest.TestCase):
    """Test single-image prediction, un-tiling, score sorting, and cap limits."""

    def setUp(self) -> None:
        self.mock_model = MagicMock()
        self.mock_model.category_names = list(CANONICAL_CLASSES.values())

    @patch("src.infer.sahi.get_sliced_prediction")
    def test_coordinate_space_and_xywh_export(self, mock_sliced_pred: MagicMock) -> None:
        # Mock SAHI output with boxes in global space
        preds = [
            make_object_prediction([100.0, 150.0, 120.0, 180.0], cat_id=0, cat_name="airplane", score=0.85),
            make_object_prediction([400.0, 500.0, 415.0, 520.0], cat_id=6, cat_name="person", score=0.92),
        ]
        mock_result = PredictionResult(
            object_prediction_list=preds,
            image=np.zeros((800, 800, 3), dtype=np.uint8),
            durations_in_seconds={"slice": 0.01, "prediction": 0.05, "postprocess": 0.005},
        )
        mock_sliced_pred.return_value = mock_result

        runner = SahiInferenceRunner(detection_model=self.mock_model)
        dummy_img = np.zeros((800, 800, 3), dtype=np.uint8)
        detections, durations = runner.predict_image(dummy_img, image_id=42)

        self.assertEqual(len(detections), 2)
        # Sorted descending by score: second pred (score 0.92) is first
        self.assertEqual(detections[0]["image_id"], 42)
        self.assertEqual(detections[0]["category_id"], CANONICAL_NAME_TO_ID["person"])
        self.assertEqual(detections[0]["bbox"], [400.0, 500.0, 15.0, 20.0])
        self.assertAlmostEqual(detections[0]["score"], 0.92)

        self.assertEqual(detections[1]["category_id"], CANONICAL_NAME_TO_ID["airplane"])
        self.assertEqual(detections[1]["bbox"], [100.0, 150.0, 20.0, 30.0])
        self.assertAlmostEqual(detections[1]["score"], 0.85)

        self.assertIn("slice_ms", durations)
        self.assertIn("prediction_ms", durations)
        self.assertIn("total_ms", durations)

    @patch("src.infer.sahi.get_sliced_prediction")
    def test_empty_predictions_handling(self, mock_sliced_pred: MagicMock) -> None:
        mock_result = PredictionResult(
            object_prediction_list=[],
            image=np.zeros((800, 800, 3), dtype=np.uint8),
            durations_in_seconds={"slice": 0.01, "prediction": 0.02, "postprocess": 0.001},
        )
        mock_sliced_pred.return_value = mock_result

        runner = SahiInferenceRunner(detection_model=self.mock_model)
        dummy_img = np.zeros((800, 800, 3), dtype=np.uint8)
        detections, durations = runner.predict_image(dummy_img, image_id=99)

        self.assertEqual(detections, [])
        self.assertGreater(durations["total_ms"], 0.0)

    @patch("src.infer.sahi.get_sliced_prediction")
    def test_max_detections_cap(self, mock_sliced_pred: MagicMock) -> None:
        # Create 10 mock predictions with ascending scores
        preds = [
            make_object_prediction(
                [10.0, 10.0, 20.0, 20.0],
                cat_id=0,
                cat_name="airplane",
                score=0.1 * i,
            )
            for i in range(1, 11)
        ]
        mock_result = PredictionResult(
            object_prediction_list=preds,
            image=np.zeros((800, 800, 3), dtype=np.uint8),
            durations_in_seconds={"slice": 0.01, "prediction": 0.02, "postprocess": 0.001},
        )
        mock_sliced_pred.return_value = mock_result

        cfg = SahiConfig(max_detections_per_image=3)
        runner = SahiInferenceRunner(detection_model=self.mock_model, config=cfg)
        detections, _ = runner.predict_image(np.zeros((800, 800, 3), dtype=np.uint8), image_id=1)

        self.assertEqual(len(detections), 3)
        # Should keep top 3 highest scores (1.0, 0.9, 0.8)
        self.assertAlmostEqual(detections[0]["score"], 1.0)
        self.assertAlmostEqual(detections[1]["score"], 0.9)
        self.assertAlmostEqual(detections[2]["score"], 0.8)


class TestExportSahiPredictions(unittest.TestCase):
    """Test full dataset split export pipeline and manifest writing."""

    def test_export_pipeline_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create dummy weights
            weights_file = tmp_path / "model.pt"
            weights_file.write_bytes(b"dummy_weights_content")

            # Create dummy image
            img_dir = tmp_path / "images"
            img_dir.mkdir()
            img_file = img_dir / "sample.png"
            Image.fromarray(np.zeros((800, 800, 3), dtype=np.uint8)).save(img_file)

            # Create dummy annotations
            ann_file = tmp_path / "ann.json"
            ann_data = {
                "images": [{"id": 101, "file_name": "sample.png", "width": 800, "height": 800}],
                "categories": [{"id": cid, "name": name} for cid, name in CANONICAL_CLASSES.items()],
                "annotations": [],
            }
            ann_file.write_text(json.dumps(ann_data), encoding="utf-8")

            # Create dummy dataset yaml
            yaml_file = tmp_path / "dataset.yaml"
            yaml_content = (
                f"path: {tmp_path.as_posix()}\n"
                f"val: images\n"
                f"names:\n" + "\n".join(f"  {k}: {v}" for k, v in CANONICAL_CLASSES.items())
            )
            yaml_file.write_text(yaml_content, encoding="utf-8")

            output_dir = tmp_path / "output"

            # Mock SahiInferenceRunner to return deterministic prediction
            with patch("src.infer.sahi.SahiInferenceRunner") as MockRunnerClass:
                mock_runner = MagicMock()
                mock_runner.predict_image.return_value = (
                    [
                        {
                            "image_id": 101,
                            "category_id": 0,
                            "bbox": [50.0, 60.0, 15.0, 12.0],
                            "score": 0.88,
                        }
                    ],
                    {
                        "slice_ms": 10.0,
                        "prediction_ms": 25.0,
                        "postprocess_ms": 5.0,
                        "total_ms": 40.0,
                    },
                )
                MockRunnerClass.return_value = mock_runner

                manifest = export_sahi_predictions(
                    weights=weights_file,
                    data=yaml_file,
                    output_dir=output_dir,
                    split="val",
                    annotations=ann_file,
                    image_root=img_dir,
                )

                pred_file = output_dir / "predictions.json"
                manifest_file = output_dir / "manifest.json"

                self.assertTrue(pred_file.is_file())
                self.assertTrue(manifest_file.is_file())

                preds = json.loads(pred_file.read_text(encoding="utf-8"))
                self.assertEqual(len(preds), 1)
                self.assertEqual(preds[0]["image_id"], 101)
                self.assertEqual(preds[0]["category_id"], 0)
                self.assertEqual(preds[0]["bbox"], [50.0, 60.0, 15.0, 12.0])

                self.assertEqual(manifest["total_images"], 1)
                self.assertEqual(manifest["total_detections"], 1)
                self.assertEqual(manifest["zero_detection_image_ids"], [])
                self.assertIn("throughput_fps", manifest["latency"])


if __name__ == "__main__":
    unittest.main()
