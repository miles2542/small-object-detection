"""Tests for the P2 (stride-4) architectures, the model builder and `sod model` commands."""

from __future__ import annotations

import unittest
from pathlib import Path

from typer.testing import CliRunner
from ultralytics import YOLO

from src.cli import app
from src.models.builder import build_model, measure_receptive_fields, verify_model

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "models"
P2_CONFIGS = {
    "yolov8s-p2": CONFIG_DIR / "yolov8s-p2.yaml",
    "yolo26s-p2": CONFIG_DIR / "yolo26s-p2.yaml",
}
EXPECTED_STRIDES = [4, 8, 16, 32]


class TestP2Architecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = {name: build_model(path) for name, path in P2_CONFIGS.items()}

    def test_config_files_exist(self) -> None:
        for path in P2_CONFIGS.values():
            self.assertTrue(path.exists(), path)

    def test_four_heads_with_expected_strides(self) -> None:
        for name, model in self.models.items():
            head = model.model.model[-1]
            with self.subTest(model=name):
                self.assertEqual(head.nl, 4)
                self.assertEqual([int(s) for s in head.stride.tolist()], EXPECTED_STRIDES)
                self.assertEqual(head.nc, 8)

    def test_feature_map_shapes_at_800(self) -> None:
        for name, model in self.models.items():
            report = verify_model(model, imgsz=800)
            with self.subTest(model=name):
                self.assertEqual(report.strides, EXPECTED_STRIDES)
                self.assertEqual(
                    [shape[-2:] for shape in report.feature_shapes],
                    [(200, 200), (100, 100), (50, 50), (25, 25)],
                )

    def test_feature_map_shapes_follow_input_size(self) -> None:
        report = verify_model(self.models["yolov8s-p2"], imgsz=640)
        self.assertEqual(
            [shape[-2:] for shape in report.feature_shapes],
            [(160, 160), (80, 80), (40, 40), (20, 20)],
        )

    def test_parameter_counts(self) -> None:
        # Small-model scale: roughly 10M parameters for both families.
        for name, model in self.models.items():
            report = verify_model(model, imgsz=800)
            with self.subTest(model=name):
                self.assertGreater(report.parameters, 8_000_000)
                self.assertLess(report.parameters, 12_000_000)
                self.assertGreater(report.gflops, 0)

    def test_end2end_flag_matches_family(self) -> None:
        self.assertFalse(verify_model(self.models["yolov8s-p2"]).end2end)
        self.assertTrue(verify_model(self.models["yolo26s-p2"]).end2end)

    def test_baseline_has_no_stride_four_head(self) -> None:
        baseline = YOLO("yolov8s.yaml")
        strides = [int(s) for s in baseline.model.model[-1].stride.tolist()]
        self.assertEqual(strides, [8, 16, 32])

    def test_num_classes_override(self) -> None:
        model = build_model(P2_CONFIGS["yolov8s-p2"], num_classes=3)
        self.assertEqual(model.model.model[-1].nc, 3)

    def test_missing_config_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            build_model(CONFIG_DIR / "does-not-exist.yaml")

    def test_receptive_fields_cover_each_scale(self) -> None:
        fields = measure_receptive_fields(self.models["yolov8s-p2"], imgsz=320)
        self.assertEqual(len(fields), 4)
        self.assertTrue(all(0 < f <= 320 for f in fields))
        self.assertEqual(fields, sorted(fields))


class TestModelCli(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_model_help(self) -> None:
        result = self.runner.invoke(app, ["model", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("info", result.output)
        self.assertIn("verify", result.output)

    def test_model_info(self) -> None:
        result = self.runner.invoke(
            app, ["model", "info", "--config", str(P2_CONFIGS["yolov8s-p2"]), "--imgsz", "320"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Detection heads", result.output)
        self.assertIn("4, 8, 16, 32", result.output)
        self.assertIn("Receptive field", result.output)

    def test_non_multiple_of_stride_is_a_clean_error(self) -> None:
        for command in ("verify", "info"):
            with self.subTest(command=command):
                result = self.runner.invoke(
                    app,
                    ["model", command, "--config", str(P2_CONFIGS["yolov8s-p2"]), "--imgsz", "810"],
                )
                self.assertEqual(result.exit_code, 1)
                self.assertIn("multiple of 32", result.output)
                self.assertNotIsInstance(result.exception, RuntimeError)

    def test_nwd_options_require_nwd_flag(self) -> None:
        for option in (["--nwd-ratio", "0.0"], ["--nwd-constant", "10"]):
            with self.subTest(option=option):
                result = self.runner.invoke(app, ["train", "run", *option])
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("--nwd", result.output)

    def test_model_verify(self) -> None:
        result = self.runner.invoke(
            app, ["model", "verify", "--config", str(P2_CONFIGS["yolo26s-p2"]), "--imgsz", "800"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        for scale in ("P2", "P3", "P4", "P5"):
            self.assertIn(scale, result.output)
        for size in ("200, 200", "100, 100", "50, 50", "25, 25"):
            self.assertIn(size, result.output)


if __name__ == "__main__":
    unittest.main()
