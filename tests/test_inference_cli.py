"""CLI integration tests for inference subcommands (sod infer sahi/run/bench)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from src.cli import app


class TestInferenceCli(unittest.TestCase):
    """Test CLI commands under sod infer."""

    def setUp(self) -> None:
        self.runner = CliRunner()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

        # Create dummy weights and YAML
        self.weights = self.root / "model.pt"
        self.weights.write_bytes(b"dummy_weights")

        self.yaml_file = self.root / "dataset.yaml"
        self.yaml_file.write_text("path: .\nval: images\nnames: {0: airplane}\n", encoding="utf-8")

    def test_infer_help_messages(self) -> None:
        result = self.runner.invoke(app, ["infer", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("sahi", result.output)
        self.assertIn("run", result.output)
        self.assertIn("bench", result.output)

        result_sahi = self.runner.invoke(app, ["infer", "sahi", "--help"])
        self.assertEqual(result_sahi.exit_code, 0)
        self.assertIn("--slice-size", result_sahi.output)
        self.assertIn("--overlap", result_sahi.output)

        result_run = self.runner.invoke(app, ["infer", "run", "--help"])
        self.assertEqual(result_run.exit_code, 0)
        self.assertIn("--img-size", result_run.output)

        result_bench = self.runner.invoke(app, ["infer", "bench", "--help"])
        self.assertEqual(result_bench.exit_code, 0)
        self.assertIn("--resolutions", result_bench.output)
        self.assertIn("--with-sahi", result_bench.output)

    @patch("src.infer.sahi.export_sahi_predictions")
    def test_infer_sahi_execution(self, mock_export: MagicMock) -> None:
        out_dir = self.root / "sahi_out"
        mock_export.return_value = {
            "total_images": 5,
            "total_detections": 25,
            "zero_detection_image_ids": [],
            "latency": {
                "mean_slice_ms": 1.2,
                "mean_prediction_ms": 15.4,
                "mean_postprocess_ms": 2.1,
                "mean_total_ms": 18.7,
                "throughput_fps": 53.4,
            },
        }

        result = self.runner.invoke(
            app,
            [
                "infer",
                "sahi",
                "--weights",
                str(self.weights),
                "--data",
                str(self.yaml_file),
                "--output-dir",
                str(out_dir),
                "--slice-size",
                "512",
                "--overlap",
                "0.20",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("SAHI Sliced Inference", result.output)
        self.assertIn("53.40 FPS", result.output)
        mock_export.assert_called_once()

    @patch("src.infer.high_res.export_high_res_predictions")
    def test_infer_run_execution(self, mock_export: MagicMock) -> None:
        out_dir = self.root / "high_res_out"
        mock_export.return_value = {
            "total_images": 10,
            "total_detections": 40,
            "zero_detection_image_ids": [],
            "latency": {
                "mean_preprocess_ms": 1.0,
                "mean_inference_ms": 12.0,
                "mean_postprocess_ms": 1.5,
                "mean_total_ms": 14.5,
                "throughput_fps": 68.9,
            },
            "memory": {"peak_mb": 120.0},
        }

        result = self.runner.invoke(
            app,
            [
                "infer",
                "run",
                "--weights",
                str(self.weights),
                "--data",
                str(self.yaml_file),
                "--output-dir",
                str(out_dir),
                "--img-size",
                "1280",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Standard High-Resolution Inference", result.output)
        self.assertIn("68.90 FPS", result.output)
        self.assertIn("120.00 MB", result.output)
        mock_export.assert_called_once()

    @patch("src.infer.high_res.benchmark_resolutions")
    def test_infer_bench_execution(self, mock_bench: MagicMock) -> None:
        out_dir = self.root / "bench_out"
        mock_bench.return_value = {
            "weights": str(self.weights),
            "device": "cpu",
            "sample_count": 10,
            "resolutions": {
                "800": {
                    "image_size": 800,
                    "latency": {"mean_total_ms": 12.0, "throughput_fps": 83.3},
                    "memory": {"peak_mb": 110.0},
                    "total_detections": 30,
                },
                "1280": {
                    "image_size": 1280,
                    "latency": {"mean_total_ms": 28.0, "throughput_fps": 35.7},
                    "memory": {"peak_mb": 160.0},
                    "total_detections": 45,
                },
            },
        }

        result = self.runner.invoke(
            app,
            [
                "infer",
                "bench",
                "--weights",
                str(self.weights),
                "--data",
                str(self.yaml_file),
                "--output-dir",
                str(out_dir),
                "--resolutions",
                "800,1280",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Inference Strategy & Latency Benchmark", result.output)
        self.assertIn("800x800", result.output)
        self.assertIn("1280x1280", result.output)
        mock_bench.assert_called_once()


if __name__ == "__main__":
    unittest.main()
