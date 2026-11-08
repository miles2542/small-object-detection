"""CLI integration and input-resolution checks with tiny real COCO fixtures."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image
from typer.testing import CliRunner

from src.cli import app
from src.eval.audit import sha256
from src.eval.runner import resolve_inputs


class EvaluationCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gt = self.root / "annotations.json"
        self.pred = self.root / "predictions.json"
        self.data = {
            "images": [{"id": 0, "file_name": "same.png", "width": 40, "height": 20}],
            "categories": [{"id": 7, "name": "person"}, {"id": 91, "name": "airplane"}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 7,
                    "bbox": [1, 1, 4, 4],
                    "area": 16,
                    "iscrowd": 0,
                }
            ],
        }
        self.gt.write_text(json.dumps(self.data))
        self.rows = [{"image_id": 0, "category_id": 7, "bbox": [1, 1, 4, 4], "score": 0.9}]
        self.pred.write_text(json.dumps(self.rows))
        self.runner = CliRunner()

    def invoke(self, *arguments):
        return self.runner.invoke(app, ["eval", "run", *map(str, arguments)])

    def test_help_saved_predictions_and_repeatability(self):
        help_result = self.invoke("--help")
        self.assertEqual(help_result.exit_code, 0)
        self.assertIn("--weights", help_result.output)
        self.assertNotIn("--protocol", help_result.output)
        for name in ["first", "second"]:
            output = self.root / name
            result = self.invoke(
                "--data", self.gt, "--predictions", self.pred, "--output-dir", output
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("N/A", result.output)
            self.assertIn("100.0000", result.output)
            record = json.loads((output / "run.json").read_text())
            self.assertEqual(record["image_count"], 1)
        first = json.loads((self.root / "first/evaluation/metrics.json").read_text())
        second = json.loads((self.root / "second/evaluation/metrics.json").read_text())
        self.assertEqual(first, second)

    def test_bad_modes_splits_protocols_and_missing_arguments(self):
        for args in [
            [],
            ["--split", "invalid"],
            ["--protocol", "unknown"],
            ["--weights", self.gt, "--predictions", self.pred],
        ]:
            base = ["--data", self.gt, "--output-dir", self.root / "bad"]
            if args and "--weights" not in args:
                base.extend(["--predictions", self.pred])
            result = self.invoke(*base, *args)
            self.assertNotEqual(result.exit_code, 0)
            self.assertFalse((self.root / "bad/run.json").exists())
        self.assertNotEqual(self.invoke("--predictions", self.pred).exit_code, 0)
        wrong_data = self.invoke(
            "--data",
            self.pred,
            "--predictions",
            self.pred,
            "--output-dir",
            self.root / "wrong_data",
        )
        self.assertNotEqual(wrong_data.exit_code, 0)
        self.assertIn("COCO JSON", wrong_data.output)

    def test_manifest_smoke_guard_and_confidence_floor(self):
        manifest = {
            "status": "complete",
            "purpose": "smoke_only",
            "predictions_sha256": sha256(self.pred),
            "annotations": {"sha256": sha256(self.gt)},
            "image_ids": [0],
            "detections": 1,
            "prediction_counts": {"0": 1},
            "predict_settings": {"conf": 0.001},
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        blocked = self.invoke(
            "--data", self.gt, "--predictions", self.pred, "--output-dir", self.root / "blocked"
        )
        self.assertNotEqual(blocked.exit_code, 0)
        allowed = self.invoke(
            "--data",
            self.gt,
            "--predictions",
            self.pred,
            "--smoke",
            "--output-dir",
            self.root / "allowed",
        )
        self.assertEqual(allowed.exit_code, 0, allowed.output)
        self.assertIn("SMOKE ONLY", allowed.output)
        below = self.invoke(
            "--data",
            self.gt,
            "--predictions",
            self.pred,
            "--smoke",
            "--confidence",
            "0.0001",
            "--output-dir",
            self.root / "below",
        )
        self.assertNotEqual(below.exit_code, 0)
        self.assertIn("export floor", below.output)

    def test_yaml_selection_and_weight_mode_orchestration(self):
        images = self.root / "images"
        images.mkdir()
        Image.new("RGB", (40, 20)).save(images / "same.png")
        config = self.root / "dataset.yaml"
        config.write_text(
            yaml.safe_dump(
                {"path": str(self.root), "val": "images", "names": {0: "person", 1: "airplane"}}
            )
        )
        _, resolved, ids = resolve_inputs(config, "val", self.gt, None)
        self.assertEqual(resolved, images.resolve())
        self.assertEqual(ids, [0])
        canonical = self.root / "annotations"
        canonical.mkdir()
        canonical = canonical / "aitod_val.json"
        canonical.write_text(json.dumps(self.data))
        discovered, _, _ = resolve_inputs(config, "val", None, None)
        self.assertEqual(discovered, canonical.resolve())
        weights = self.root / "fixture.pt"
        weights.write_bytes(b"not loaded; export mocked at model boundary")
        early = self.invoke(
            "--data",
            config,
            "--weights",
            weights,
            "--annotations",
            self.gt,
            "--confidence",
            "0.0001",
            "--output-dir",
            self.root / "early",
        )
        self.assertNotEqual(early.exit_code, 0)
        self.assertFalse((self.root / "early").exists())

        def fake_export(weights, annotations, image_root, output_dir, **kwargs):
            output_dir.mkdir()
            (output_dir / "predictions.json").write_text(json.dumps(self.rows))
            (output_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "purpose": "evaluation",
                        "image_ids": [0],
                        "detections": 1,
                        "prediction_counts": {"0": 1},
                        "annotations": {"sha256": sha256(annotations)},
                        "predictions_sha256": sha256(output_dir / "predictions.json"),
                    }
                )
            )

        with patch("src.eval.export.export_predictions", side_effect=fake_export):
            output = self.root / "weighted"
            result = self.invoke(
                "--data",
                config,
                "--weights",
                weights,
                "--annotations",
                self.gt,
                "--output-dir",
                output,
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue((output / "ground_truth.json").exists())
        self.assertEqual(json.loads((output / "run.json").read_text())["mode"], "weights")
        Image.new("RGB", (40, 20)).save(images / "unknown.png")
        with self.assertRaisesRegex(ValueError, "unknown files"):
            resolve_inputs(config, "val", self.gt, None)
        config.write_text("names: [broken")
        with self.assertRaisesRegex(ValueError, "Invalid dataset YAML"):
            resolve_inputs(config, "val", self.gt, None)

    def test_compare_cli(self):
        help_result = self.runner.invoke(app, ["eval", "compare", "--help"])
        self.assertEqual(help_result.exit_code, 0)
        self.assertIn("--run", help_result.output)
        self.assertIn("--category-id", help_result.output)
        self.assertIn("--reference-bin", help_result.output)

        # Create two valid runs
        r1 = self.root / "r1"
        r2 = self.root / "r2"
        self.invoke("--data", self.gt, "--predictions", self.pred, "--output-dir", r1)
        self.invoke("--data", self.gt, "--predictions", self.pred, "--output-dir", r2)

        cmp_out = self.root / "cmp"
        with patch("src.eval.plots.plot_analysis_comparison"):
            res = self.runner.invoke(
                app,
                [
                    "eval",
                    "compare",
                    "--run",
                    f"m1={r1}",
                    "--run",
                    f"m2={r2}",
                    "--output-dir",
                    str(cmp_out),
                    "--category-id",
                    "7",
                ],
            )
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Model Comparison Across Scale Bins", res.output)
        self.assertIn("Class-Controlled Scale Gaps", res.output)
        self.assertTrue((cmp_out / "analysis.json").exists())
        self.assertTrue((cmp_out / "comparison.md").exists())

        # Bad formats and single run
        bad_run = self.runner.invoke(app, ["eval", "compare", "--run", "invalid_format"])
        self.assertNotEqual(bad_run.exit_code, 0)

        dup_run = self.runner.invoke(
            app, ["eval", "compare", "--run", f"m1={r1}", "--run", f"m1={r2}"]
        )
        self.assertNotEqual(dup_run.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
