"""Tests for injecting the hybrid CIoU/NWD box loss into Ultralytics criteria."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import BboxLoss

from src.losses.nwd import nwd_score
from src.models.builder import build_model
from src.models.loss_bridge import NWDBboxLoss, apply_nwd_loss, make_nwd_trainer

REG_MAX = 16


def _bbox_loss_inputs(stride_value: float = 4.0):
    """One image, six anchors, three foreground, with pixel-unit boxes of 8-14 px."""
    torch.manual_seed(0)
    anchors = 6
    stride = torch.full((anchors, 1), stride_value)
    # Boxes in feature-grid units (pixels / stride), as Ultralytics passes them.
    pred = torch.tensor(
        [[10.0, 10.0, 13.0, 13.5], [20.0, 20.0, 22.0, 23.0], [5.0, 5.0, 7.5, 7.0]]
    )
    target = torch.tensor(
        [[10.5, 10.0, 13.5, 13.0], [21.0, 20.5, 23.5, 22.0], [5.0, 5.5, 7.0, 7.5]]
    )
    pred_bboxes = torch.zeros(1, anchors, 4)
    target_bboxes = torch.zeros(1, anchors, 4)
    pred_bboxes[0, :3], target_bboxes[0, :3] = pred, target
    fg_mask = torch.tensor([[True, True, True, False, False, False]])
    target_scores = torch.zeros(1, anchors, 2)
    target_scores[0, :3, 0] = torch.tensor([0.9, 0.5, 0.7])
    return {
        "pred_dist": torch.randn(1, anchors, 4 * REG_MAX),
        "pred_bboxes": pred_bboxes,
        "anchor_points": torch.rand(anchors, 2) * 10,
        "target_bboxes": target_bboxes,
        "target_scores": target_scores,
        "target_scores_sum": target_scores.sum().clamp(min=1),
        "fg_mask": fg_mask,
        "imgsz": torch.tensor([320.0, 320.0]),
        "stride": stride,
    }


class TestNWDBboxLoss(unittest.TestCase):
    def test_alpha_one_matches_ultralytics_ciou_term(self) -> None:
        inputs = _bbox_loss_inputs()
        expected_box, expected_dfl = BboxLoss(REG_MAX)(**inputs)
        box, dfl = NWDBboxLoss(REG_MAX, alpha=1.0)(**inputs)
        torch.testing.assert_close(box, expected_box)
        torch.testing.assert_close(dfl, expected_dfl)

    def test_alpha_zero_uses_nwd_in_pixel_units(self) -> None:
        inputs = _bbox_loss_inputs(stride_value=4.0)
        box, _ = NWDBboxLoss(REG_MAX, alpha=0.0, constant=12.8)(**inputs)

        fg = inputs["fg_mask"][0]
        scale = 4.0
        pred = inputs["pred_bboxes"][0][fg] * scale
        target = inputs["target_bboxes"][0][fg] * scale
        weight = inputs["target_scores"][0][fg].sum(-1)
        # xyxy -> xywh in pixels
        def xywh(b: torch.Tensor) -> torch.Tensor:
            return torch.cat(((b[:, :2] + b[:, 2:]) / 2, b[:, 2:] - b[:, :2]), dim=1)

        expected = ((1 - nwd_score(xywh(pred), xywh(target), 12.8)) * weight).sum()
        expected = expected / inputs["target_scores_sum"]
        torch.testing.assert_close(box, expected)

    def test_constant_is_pixel_based_not_grid_based(self) -> None:
        # The same pixel-space boxes must give the same loss regardless of the stride they use.
        a = _bbox_loss_inputs(stride_value=4.0)
        b = _bbox_loss_inputs(stride_value=8.0)
        for key in ("pred_bboxes", "target_bboxes"):
            b[key] = a[key] / 2  # same pixel boxes expressed at stride 8
        loss_a, _ = NWDBboxLoss(REG_MAX, alpha=0.0)(**a)
        loss_b, _ = NWDBboxLoss(REG_MAX, alpha=0.0)(**b)
        torch.testing.assert_close(loss_a, loss_b)

    def test_empty_foreground_is_zero_and_differentiable(self) -> None:
        inputs = _bbox_loss_inputs()
        inputs["fg_mask"] = torch.zeros_like(inputs["fg_mask"])
        inputs["pred_bboxes"] = inputs["pred_bboxes"].clone().requires_grad_()
        box, _ = NWDBboxLoss(REG_MAX)(**inputs)
        self.assertEqual(box.item(), 0.0)
        box.backward()

    def test_gradient_flows_to_predicted_boxes(self) -> None:
        inputs = _bbox_loss_inputs()
        inputs["pred_bboxes"] = inputs["pred_bboxes"].clone().requires_grad_()
        box, _ = NWDBboxLoss(REG_MAX, alpha=0.0)(**inputs)
        box.backward()
        grad = inputs["pred_bboxes"].grad
        self.assertTrue(torch.isfinite(grad).all().item())
        self.assertGreater(grad[0, :3].abs().sum().item(), 0.0)


class TestApplyNwdLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v8 = build_model("configs/models/yolov8s-p2.yaml").model
        cls.e2e = build_model("configs/models/yolo26s-p2.yaml").model
        for net in (cls.v8, cls.e2e):
            net.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=10)

    def test_replaces_box_loss_for_v8(self) -> None:
        criterion = apply_nwd_loss(self.v8.init_criterion(), alpha=0.3, constant=9.0)
        self.assertIsInstance(criterion.bbox_loss, NWDBboxLoss)
        self.assertEqual(criterion.bbox_loss.box_loss.alpha, 0.3)
        self.assertEqual(criterion.bbox_loss.box_loss.constant, 9.0)

    def test_replaces_both_branches_for_end_to_end(self) -> None:
        criterion = apply_nwd_loss(self.e2e.init_criterion())
        self.assertIsInstance(criterion.one2many.bbox_loss, NWDBboxLoss)
        self.assertIsInstance(criterion.one2one.bbox_loss, NWDBboxLoss)

    def test_rejects_criterion_without_box_loss(self) -> None:
        with self.assertRaises(TypeError):
            apply_nwd_loss(object())

    def test_full_criterion_changes_only_box_term(self) -> None:
        batch = {
            "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
            "cls": torch.tensor([[0.0], [5.0], [3.0]]),
            "bboxes": torch.tensor(
                [[0.3, 0.3, 0.05, 0.05], [0.6, 0.5, 0.04, 0.08], [0.5, 0.5, 0.03, 0.03]]
            ),
        }
        for name, net in (("v8", self.v8), ("e2e", self.e2e)):
            with self.subTest(model=name):
                net.train()
                torch.manual_seed(0)
                preds = net(torch.randn(2, 3, 320, 320))
                base_loss, _ = net.loss(batch, preds)
                net.criterion = apply_nwd_loss(net.init_criterion(), alpha=0.0)
                nwd_loss, _ = net.loss(batch, preds)
                net.criterion = None

                self.assertTrue(torch.isfinite(nwd_loss).all().item())
                self.assertNotEqual(base_loss[0].item(), nwd_loss[0].item())
                torch.testing.assert_close(base_loss[1:], nwd_loss[1:])


class TestMakeNwdTrainer(unittest.TestCase):
    def test_trainer_carries_hyperparameters(self) -> None:
        trainer_cls = make_nwd_trainer(nwd_ratio=0.25, nwd_constant=16.0)
        self.assertTrue(issubclass(trainer_cls, DetectionTrainer))
        self.assertEqual(trainer_cls.alpha, 0.25)
        self.assertEqual(trainer_cls.constant, 16.0)

    def test_invalid_ratio_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            make_nwd_trainer(nwd_ratio=1.5)


if __name__ == "__main__":
    unittest.main()
