"""Export contract tests with real Results objects and deterministic model fixtures."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from ultralytics.engine.results import Results
from ultralytics.utils.ops import scale_boxes

from src.eval.export import category_mapping, export_predictions, results_to_coco

NAMES = {0: "person", 4: "airplane"}
CATEGORIES = [{"id": 17, "name": "airplane"}, {"id": 7, "name": "person"}]


def result(path="42.png", boxes=None):
    values = [[64, 32, 192, 96, 0.75, 4]] if boxes is None else boxes
    return Results(
        np.zeros((320, 640, 3), dtype=np.uint8),
        path=str(path),
        names=NAMES,
        boxes=torch.tensor(values, dtype=torch.float32).reshape(-1, 6),
    )


class FakeModel:
    names, task = NAMES, "detect"

    def __init__(self, omit=False, empty=False):
        self.omit, self.empty = omit, empty
        self.model = SimpleNamespace(model=[SimpleNamespace(end2end=False)])
        self.predictor = SimpleNamespace(device="cpu", model=SimpleNamespace(model=self.model))

    def predict(self, source, **kwargs):
        for i, path in enumerate(source):
            if not (self.omit and i == 1):
                yield result(path, [] if self.empty or i else None)


class PredictionExportTests(unittest.TestCase):
    def test_exact_names_and_noncontiguous_ids(self):
        self.assertEqual(category_mapping(NAMES, CATEGORIES), {0: 7, 4: 17})
        with self.assertRaises(ValueError):
            category_mapping({0: "car"}, CATEGORIES)
        self.assertEqual(category_mapping({0: "person", 1: "car"}, CATEGORIES, smoke=True), {0: 7})
        with self.assertRaises(ValueError):
            category_mapping({0: "person", 1: "person"}, CATEGORIES, smoke=True)

    def test_original_pixels_after_rectangular_letterbox(self):
        # 640x320 -> 800x800: gain 1.25, vertical padding 200 pixels.
        boxes = scale_boxes((800, 800), torch.tensor([[80.0, 240.0, 240.0, 320.0]]), (320, 640))
        sample = result(boxes=[[*boxes[0].tolist(), 0.75, 4]])
        rows = results_to_coco(sample, {"id": 91, "width": 640, "height": 320}, {4: 17})
        self.assertEqual(
            rows,
            [{"image_id": 91, "category_id": 17, "bbox": [64.0, 32.0, 128.0, 64.0], "score": 0.75}],
        )

    def test_invalid_results_are_rejected(self):
        for values in [
            [64, 32, 64, 96, 0.75, 4],
            [-1, 32, 192, 96, 0.75, 4],
            [64, 32, 192, 96, float("nan"), 4],
            [64, 32, 192, 96, 1.2, 4],
            [64, 32, 192, 96, 0.75, 2],
        ]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                results_to_coco(
                    result(boxes=[values]), {"id": 91, "width": 640, "height": 320}, {4: 17}
                )

    def test_manifest_keeps_empty_images_and_duplicate_stems(self):
        self.run_fixture(empty=False)
        self.run_fixture(empty=True)

    def run_fixture(self, empty):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sub in ["a", "b"]:
                (root / sub).mkdir()
                Image.new("RGB", (640, 320)).save(root / sub / "42.png")
            annotations = root / "gt.json"
            annotations.write_text(
                json.dumps(
                    {
                        "categories": CATEGORIES,
                        "images": [
                            {"id": 91, "file_name": "a/42.png", "width": 640, "height": 320},
                            {"id": 0, "file_name": "b/42.png", "width": 640, "height": 320},
                        ],
                    }
                )
            )
            weights = root / "fixture.pt"
            weights.write_bytes(b"model fixture; never loaded by torch")
            output = root / "export"
            with patch("ultralytics.YOLO", return_value=FakeModel(empty=empty)):
                manifest = export_predictions(weights, annotations, root, output)
            rows = json.loads((output / "predictions.json").read_text())
            self.assertEqual(manifest["image_ids"], [91, 0])
            self.assertEqual(manifest["zero_detection_image_ids"], [91, 0] if empty else [0])
            self.assertEqual(
                rows,
                []
                if empty
                else [
                    {
                        "image_id": 91,
                        "category_id": 17,
                        "bbox": [64.0, 32.0, 128.0, 64.0],
                        "score": 0.75,
                    }
                ],
            )
            with self.assertRaises(FileExistsError):
                export_predictions(weights, annotations, root, output)
            with (
                patch("ultralytics.YOLO", return_value=FakeModel(omit=True)),
                self.assertRaises(ValueError),
            ):
                export_predictions(weights, annotations, root, root / "missing")
            self.assertFalse((root / "missing/predictions.json.part").exists())
            self.assertFalse((root / "missing/manifest.json").exists())
            limited = FakeModel()
            limited.model.model[-1] = SimpleNamespace(end2end=True, max_det=300)
            with (
                patch("ultralytics.YOLO", return_value=limited),
                self.assertRaisesRegex(ValueError, "head truncates"),
            ):
                export_predictions(weights, annotations, root, root / "limited")
            self.assertFalse((root / "limited/manifest.json").exists())
            inactive = FakeModel()
            inactive.model.model[-1].one2one_cv2 = object()
            with (
                patch("ultralytics.YOLO", return_value=inactive),
                self.assertRaisesRegex(ValueError, "was not enabled"),
            ):
                export_predictions(weights, annotations, root, root / "inactive")
            self.assertFalse((root / "inactive/manifest.json").exists())
            copied = FakeModel()
            copied.model.model[-1].one2one_cv2 = object()
            copied.predictor.model.model = SimpleNamespace(
                model=[SimpleNamespace(end2end=True, max_det=1500)]
            )
            with patch("ultralytics.YOLO", return_value=copied):
                native = export_predictions(weights, annotations, root, root / "native")
            self.assertEqual(native["head_max_dets"], [1500])
            self.assertIs(native["predict_settings"]["nms"], False)
            self.assertEqual(native["postprocessing"], "native_end_to_end")


if __name__ == "__main__":
    unittest.main()
