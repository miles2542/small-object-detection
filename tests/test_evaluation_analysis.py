"""Signed-gap, provenance and match-based failure-selection checks."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.eval.analysis import analyze_runs, ap_gap, failure_candidates
from src.eval.coco_eval import evaluate_coco
from src.eval.runner import run_evaluation


class EvaluationAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.coco = {
            "images": [{"id": 7, "file_name": "test.png", "width": 100, "height": 100}],
            "categories": [{"id": 6, "name": "person"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 7,
                    "category_id": 6,
                    "bbox": [10, 10, 4, 4],
                    "area": 16,
                    "iscrowd": 0,
                }
            ],
        }
        self.gt, self.pred = self.root / "gt.json", self.root / "predictions.json"
        self.gt.write_text(json.dumps(self.coco))
        self.rows = [{"image_id": 7, "category_id": 6, "bbox": [12, 10, 4, 4], "score": 0.8}]
        self.pred.write_text(json.dumps(self.rows))

    def test_signed_and_undefined_gaps(self):
        self.assertAlmostEqual(ap_gap(0.1, 0.4)["gap_pp"], 30)
        self.assertAlmostEqual(ap_gap(0.1, 0.4)["relative_loss_percent"], 75)
        self.assertLess(ap_gap(0.6, 0.4)["gap_pp"], 0)
        self.assertIsNone(ap_gap(0.1, 0)["relative_loss_percent"])
        self.assertIsNone(ap_gap(None, 0.4)["gap_pp"])

    def test_match_selection_and_confidence(self):
        _, evaluator = evaluate_coco(self.coco, self.rows)
        candidates = failure_candidates(evaluator, 6, 0.25)
        self.assertEqual(candidates[0]["kind"], "localization_candidate")
        self.assertAlmostEqual(candidates[0]["best_same_class_iou"], 1 / 3)
        perfect = [{**self.rows[0], "bbox": [10, 10, 4, 4], "score": 0.1}]
        _, evaluator = evaluate_coco(self.coco, perfect)
        self.assertEqual(failure_candidates(evaluator, 6, 0.25)[0]["kind"], "missed")
        self.assertEqual(failure_candidates(evaluator, 6, 0.05), [])

    def test_reproduction_and_comparison_guards(self):
        runs = {name: self.root / name for name in ["one", "two"]}
        for path in runs.values():
            run_evaluation(self.gt, path, predictions=self.pred)
        with patch("src.eval.plots.plot_analysis_comparison"):
            result = analyze_runs(runs, self.root / "analysis")
        self.assertFalse(result["contains_smoke"])
        self.assertFalse(result["all_runs_verified_exports"])
        self.assertEqual(result["status"], "training_provenance_review_required")
        self.assertEqual(result["models"]["one"]["confidence_sweep"], [])
        # Same image count is insufficient: the exact protocol must agree.
        path = runs["two"] / "evaluation/metrics.json"
        metrics = json.loads(path.read_text())
        metrics["protocol"]["area_ranges"][1] = [0, 50]
        path.write_text(json.dumps(metrics))
        with self.assertRaisesRegex(ValueError, "identical"):
            analyze_runs(runs, self.root / "bad_protocol")
        self.pred.write_text("[]")
        with self.assertRaisesRegex(ValueError, "changed"):
            analyze_runs(runs, self.root / "bad_hash")

    def test_export_floor_comparison_guard(self):
        runs = {name: self.root / name for name in ["one", "two"]}
        for index, path in enumerate(runs.values()):
            run_evaluation(self.gt, path, predictions=self.pred)
            metrics_path = path / "evaluation/metrics.json"
            metrics = json.loads(metrics_path.read_text())
            provenance = metrics["provenance"]
            provenance["export_manifest"] = {
                "status": "complete",
                "annotations": {"sha256": provenance["annotations_sha256"]},
                "predictions_sha256": provenance["predictions_sha256"],
                "category_mapping": {"0": {"category_id": 6}},
                "predict_settings": {
                    "imgsz": 800,
                    "conf": [0.001, 0.1][index],
                    "max_det": 1500,
                    "augment": False,
                    "rect": False,
                    "agnostic_nms": False,
                    "iou": 0.7,
                },
            }
            metrics_path.write_text(json.dumps(metrics))
        with self.assertRaisesRegex(ValueError, "floor"):
            analyze_runs(runs, self.root / "bad_floor")


if __name__ == "__main__":
    unittest.main()
