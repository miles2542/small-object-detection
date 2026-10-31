"""Analytic detection cases and direct COCO tensor parity; no model downloads."""

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path

import numpy as np
from pycocotools.cocoeval import COCOeval

from src.eval.audit import sha256
from src.eval.coco_eval import evaluate_coco, evaluate_files


def fixture(boxes=None):
    boxes = (
        [[10, 10, 4, 4], [30, 30, 12, 12], [60, 60, 24, 24], [100, 100, 48, 48]]
        if boxes is None
        else boxes
    )
    return {
        "images": [{"id": 0, "file_name": "unrelated.png", "width": 800, "height": 800}],
        "categories": [{"id": 7, "name": "present"}, {"id": 91, "name": "absent"}],
        "annotations": [
            {
                "id": i,
                "image_id": 0,
                "category_id": 7,
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 0,
            }
            for i, box in enumerate(boxes)
        ],
    }


def predictions(coco, score=0.9):
    return [
        {
            "image_id": a["image_id"],
            "category_id": a["category_id"],
            "bbox": a["bbox"].copy(),
            "score": score,
        }
        for a in coco["annotations"]
    ]


class CocoEvaluationTests(unittest.TestCase):
    def test_perfect_empty_and_unsupported_slices(self):
        gt = fixture()
        original = deepcopy(gt)
        result, _ = evaluate_coco(gt, predictions(gt))
        for label in ["all", "verytiny", "tiny", "small", "medium"]:
            self.assertAlmostEqual(result["bins"][label]["AP"], 1)
            self.assertAlmostEqual(result["bins"][label]["AR"], 1)
            self.assertEqual(result["bins"][label]["operating_point"]["micro"]["recall"][0], 1)
        self.assertIsNone(result["per_class"]["91"]["bins"]["all"]["AP"])
        self.assertEqual(gt, original)
        empty, _ = evaluate_coco(gt, [])
        self.assertEqual(empty["bins"]["all"]["AP"], 0)
        self.assertIsNone(empty["bins"]["all"]["operating_point"]["micro"]["precision"][0])
        self.assertEqual(empty["bins"]["all"]["operating_point"]["macro"]["precision"][0], 0)
        unsupported, _ = evaluate_coco(fixture([]), [])
        self.assertIsNone(unsupported["bins"]["all"]["AP"])

    def test_missed_images_wrong_class_and_duplicate(self):
        gt = fixture([[10, 10, 4, 4]])
        gt["images"].append({"id": 52, "file_name": "52.png", "width": 800, "height": 800})
        ann = deepcopy(gt["annotations"][0])
        ann.update(id=88, image_id=52)
        gt["annotations"].append(ann)
        rows = predictions(gt)[:1]
        result, _ = evaluate_coco(gt, rows)
        self.assertEqual(result["bins"]["all"]["AR"], 0.5)
        self.assertAlmostEqual(result["bins"]["all"]["AP"], 51 / 101)
        wrong = deepcopy(rows)
        wrong[0]["category_id"] = 91
        result, _ = evaluate_coco(gt, wrong)
        point = result["bins"]["all"]["operating_point"]
        self.assertEqual(result["bins"]["all"]["AP"], 0)
        self.assertEqual(point["micro"]["fp"][0], 1)
        self.assertIsNone(point["per_class"]["91"]["precision"][0])
        duplicate = rows + deepcopy(rows)
        duplicate[1]["score"] = 0.8
        result, _ = evaluate_coco(gt, duplicate, image_ids=[0])
        self.assertAlmostEqual(result["bins"]["all"]["AP"], 1)
        self.assertEqual(result["bins"]["all"]["operating_point"]["micro"]["precision"][0], 0.5)

    def test_score_threshold_localization_and_detection_cap(self):
        gt = fixture([[10, 10, 4, 4]])
        low, _ = evaluate_coco(gt, predictions(gt, score=0.1))
        self.assertAlmostEqual(low["bins"]["all"]["AP"], 1)
        self.assertEqual(low["bins"]["all"]["operating_point"]["micro"]["recall"][0], 0)
        shifted = predictions(gt)
        shifted[0]["bbox"][0] += 2
        result, _ = evaluate_coco(gt, shifted)
        self.assertEqual(result["bins"]["all"]["AP50"], 0)
        false = deepcopy(predictions(gt)[0])
        false.update(bbox=[100, 100, 4, 4], score=0.95)
        capped, _ = evaluate_coco(gt, [false] + predictions(gt), max_detections=1)
        uncapped, _ = evaluate_coco(gt, [false] + predictions(gt), max_detections=2)
        self.assertEqual(capped["bins"]["all"]["AR"], 0)
        self.assertEqual(uncapped["bins"]["all"]["AR"], 1)

    def test_crowd_and_area_ignore(self):
        gt = fixture([[10, 10, 4, 4], [100, 100, 48, 48]])
        crowd = {
            "id": 30,
            "image_id": 0,
            "category_id": 7,
            "bbox": [200, 200, 20, 20],
            "area": 400,
            "iscrowd": 1,
        }
        gt["annotations"].append(crowd)
        rows = predictions(gt) + [
            {"image_id": 0, "category_id": 7, "bbox": [300, 300, 48, 48], "score": 0.8}
        ]
        result, _ = evaluate_coco(gt, rows)
        tiny = result["bins"]["verytiny"]
        self.assertEqual(tiny["support"], 1)
        self.assertEqual(tiny["operating_point"]["micro"]["fp"][0], 0)
        self.assertEqual(result["bins"]["all"]["support"], 2)
        self.assertEqual(result["bins"]["all"]["operating_point"]["micro"]["fp"][0], 1)

    def test_boundary_support_and_standard_coco_parity(self):
        gt = fixture(
            [[10, 10, 2, 2], [20, 20, 8, 8], [40, 40, 16, 16], [80, 80, 32, 32], [120, 120, 64, 64]]
        )
        result, actual = evaluate_coco(gt, predictions(gt))
        self.assertEqual(
            [result["bins"][b]["support"] for b in ["verytiny", "tiny", "small", "medium"]],
            [2, 2, 2, 2],
        )
        with redirect_stdout(io.StringIO()):
            standard = COCOeval(actual.cocoGt, actual.cocoDt, "bbox")
            standard.params.maxDets = actual.params.maxDets.copy()
            standard.evaluate()
            standard.accumulate()
        np.testing.assert_array_equal(actual.params.iouThrs, standard.params.iouThrs)
        np.testing.assert_array_equal(actual.params.recThrs, standard.params.recThrs)
        np.testing.assert_allclose(
            actual.eval["precision"][:, :, :, 0, :],
            standard.eval["precision"][:, :, :, 0, :],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual.eval["recall"][:, :, 0, :], standard.eval["recall"][:, :, 0, :], atol=1e-12
        )

    def test_operating_point_agrees_with_fresh_thresholded_matching(self):
        gt = fixture()
        rows = predictions(gt)
        rows[0]["score"] = 0.1
        duplicate = deepcopy(rows[1])
        duplicate["score"] = 0.3
        rows.append(duplicate)
        all_scores, _ = evaluate_coco(gt, rows)
        filtered, _ = evaluate_coco(gt, [r for r in rows if r["score"] >= 0.25])
        for label in all_scores["bins"]:
            self.assertEqual(
                all_scores["bins"][label]["operating_point"],
                filtered["bins"][label]["operating_point"],
            )

    def test_source_area_at_adjacent_float_boundaries(self):
        cases = [
            (4, [[], ["verytiny"], ["verytiny"]]),
            (64, [["verytiny"], ["verytiny", "tiny"], ["tiny"]]),
            (256, [["tiny"], ["tiny", "small"], ["small"]]),
            (1024, [["small"], ["small", "medium"], ["medium"]]),
            (4096, [["medium"], ["medium"], []]),
        ]
        for boundary, expected in cases:
            side = math.sqrt(boundary)
            for area, eligible in zip(
                [math.nextafter(boundary, -math.inf), boundary, math.nextafter(boundary, math.inf)],
                expected,
            ):
                gt = fixture([[10, 10, side, side]])
                gt["annotations"][0]["area"] = area
                metrics, _ = evaluate_coco(gt, predictions(gt))
                for label in ["verytiny", "tiny", "small", "medium"]:
                    self.assertEqual(metrics["bins"][label]["support"], int(label in eligible))

    def test_bad_inputs_and_ignored_flags_fail_clearly(self):
        gt = fixture()
        for change in [
            {"image_id": 999},
            {"category_id": 999},
            {"score": float("nan")},
            {"bbox": [0, 0, 0, 1]},
            {"score": 2},
        ]:
            rows = predictions(gt)
            rows[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate_coco(gt, rows)
        with self.assertRaises(ValueError):
            evaluate_coco(gt, [], image_ids=[])
        gt["annotations"][0]["ignore"] = 1
        with self.assertRaisesRegex(ValueError, "Standalone ignore"):
            evaluate_coco(gt, [])

    def test_saved_arrays_and_json_null_without_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gt.json").write_text(json.dumps(fixture()))
            (root / "predictions.json").write_text("[]")
            result = evaluate_files(root / "gt.json", root / "predictions.json", root / "output")
            saved = json.loads((root / "output/metrics.json").read_text())
            self.assertEqual(saved, result)
            with np.load(root / "output/curves.npz") as arrays:
                self.assertEqual(arrays["precision"].shape, (10, 101, 2, 5, 3))
            self.assertIsNone(saved["per_class"]["91"]["bins"]["all"]["AP"])
            with self.assertRaises(FileExistsError):
                evaluate_files(root / "gt.json", root / "predictions.json", root / "output")

    def test_smoke_manifest_guard_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt, pred = root / "gt.json", root / "predictions.json"
            gt.write_text(json.dumps(fixture()))
            pred.write_text("[]")
            manifest = {
                "status": "complete",
                "purpose": "smoke_only",
                "annotations": {"sha256": sha256(gt)},
                "predictions_sha256": sha256(pred),
                "image_ids": [0],
                "detections": 0,
                "prediction_counts": {"0": 0},
            }
            path = root / "manifest.json"
            for incomplete in [{}, None, []]:
                path.write_text(json.dumps(incomplete))
                with self.assertRaisesRegex(ValueError, "Incomplete export"):
                    evaluate_files(gt, pred, root / "incomplete", allow_smoke=True)
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "allow_smoke"):
                evaluate_files(gt, pred, root / "blocked")
            result = evaluate_files(gt, pred, root / "allowed", allow_smoke=True)
            self.assertEqual(result["provenance"]["purpose"], "smoke_only")
            self.assertEqual(result["bins"]["all"]["support"], 4)
            pred.write_text("[ ]")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                evaluate_files(gt, pred, root / "tampered", allow_smoke=True)


if __name__ == "__main__":
    unittest.main()
