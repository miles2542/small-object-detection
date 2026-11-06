"""Unit tests for the Normalized Gaussian Wasserstein Distance loss."""

from __future__ import annotations

import math
import unittest

import torch
from ultralytics.utils.metrics import bbox_iou

from src.losses.nwd import (
    HybridBoxLoss,
    NWDCoordinateLoss,
    nwd_score,
    wasserstein_distance_2,
)


def _boxes(*rows: tuple[float, float, float, float]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


class TestWassersteinDistance(unittest.TestCase):
    def test_identical_boxes_have_zero_distance(self) -> None:
        b = _boxes((10, 10, 12, 12), (3, 4, 5, 6))
        self.assertTrue(torch.equal(wasserstein_distance_2(b, b), torch.zeros(2)))

    def test_center_shift_matches_closed_form(self) -> None:
        # Only centers differ: W2^2 = dx^2 + dy^2 = 2.
        d = wasserstein_distance_2(_boxes((10, 10, 12, 12)), _boxes((11, 11, 12, 12)))
        self.assertAlmostEqual(d.item(), 2.0, places=6)

    def test_size_difference_matches_closed_form(self) -> None:
        # Only sizes differ: W2^2 = ((dw)^2 + (dh)^2) / 4 = (4 + 4) / 4 = 2.
        d = wasserstein_distance_2(_boxes((0, 0, 10, 10)), _boxes((0, 0, 12, 12)))
        self.assertAlmostEqual(d.item(), 2.0, places=6)

    def test_matches_general_gaussian_formula(self) -> None:
        # Diagonal covariances commute, so the matrix trace formula reduces to the closed form.
        a = _boxes((1.0, 2.0, 6.0, 10.0))
        b = _boxes((4.0, -1.0, 8.0, 4.0))
        sa = torch.diag(torch.tensor([36.0, 100.0]) / 4)
        sb = torch.diag(torch.tensor([64.0, 16.0]) / 4)
        sa_half = torch.diag(sa.diagonal().sqrt())
        cross = torch.linalg.eigh(sa_half @ sb @ sa_half)
        cross_sqrt = cross.eigenvectors @ torch.diag(cross.eigenvalues.sqrt()) @ cross.eigenvectors.T
        expected = 9.0 + 9.0 + torch.trace(sa + sb - 2 * cross_sqrt)
        self.assertAlmostEqual(
            wasserstein_distance_2(a, b).item(), expected.item(), places=4
        )

    def test_symmetric_and_translation_invariant(self) -> None:
        a = _boxes((10, 20, 6, 9))
        b = _boxes((13, 18, 8, 5))
        shift = _boxes((100, -50, 0, 0))
        self.assertAlmostEqual(
            wasserstein_distance_2(a, b).item(), wasserstein_distance_2(b, a).item(), places=6
        )
        self.assertAlmostEqual(
            wasserstein_distance_2(a, b).item(),
            wasserstein_distance_2(a + shift, b + shift).item(),
            places=3,
        )

    def test_axis_swap_invariant(self) -> None:
        a = _boxes((10, 20, 6, 9))
        b = _boxes((13, 18, 8, 5))
        swap = [1, 0, 3, 2]
        self.assertAlmostEqual(
            wasserstein_distance_2(a, b).item(),
            wasserstein_distance_2(a[:, swap], b[:, swap]).item(),
            places=6,
        )

    def test_xyxy_matches_xywh(self) -> None:
        xywh_a, xywh_b = _boxes((10, 10, 12, 8)), _boxes((13, 9, 10, 10))
        xyxy_a = _boxes((4, 6, 16, 14))
        xyxy_b = _boxes((8, 4, 18, 14))
        self.assertAlmostEqual(
            wasserstein_distance_2(xywh_a, xywh_b).item(),
            wasserstein_distance_2(xyxy_a, xyxy_b, box_format="xyxy").item(),
            places=5,
        )

    def test_invalid_format_raises(self) -> None:
        b = _boxes((0, 0, 1, 1))
        with self.assertRaises(ValueError):
            wasserstein_distance_2(b, b, box_format="cxcywh")  # type: ignore[arg-type]


class TestNwdScore(unittest.TestCase):
    def test_identical_boxes_score_one(self) -> None:
        b = _boxes((10, 10, 12, 12))
        self.assertAlmostEqual(nwd_score(b, b).item(), 1.0, places=5)

    def test_shifted_boxes_match_closed_form(self) -> None:
        b1, b2 = _boxes((10, 10, 12, 12)), _boxes((11, 11, 12, 12))
        expected = math.exp(-math.sqrt(2.0) / 12.8)
        self.assertAlmostEqual(nwd_score(b1, b2).item(), expected, places=5)

    def test_custom_constant(self) -> None:
        b1, b2 = _boxes((10, 10, 12, 12)), _boxes((11, 11, 12, 12))
        expected = math.exp(-math.sqrt(2.0) / 20.0)
        self.assertAlmostEqual(nwd_score(b1, b2, constant=20.0).item(), expected, places=5)

    def test_non_positive_constant_raises(self) -> None:
        b = _boxes((0, 0, 1, 1))
        with self.assertRaises(ValueError):
            nwd_score(b, b, constant=0.0)

    def test_non_overlapping_boxes_keep_gradient(self) -> None:
        target = _boxes((100, 0, 4, 4))
        pred = _boxes((0, 0, 4, 4)).requires_grad_()
        iou = bbox_iou(pred.detach(), target, xywh=True)
        self.assertEqual(iou.item(), 0.0)

        score = nwd_score(pred, target)
        self.assertGreater(score.item(), 0.0)
        self.assertLess(score.item(), 1.0)
        score.backward()
        # Moving the prediction toward the target (+x) must increase similarity.
        self.assertGreater(pred.grad[0, 0].item(), 0.0)

    def test_score_decreases_monotonically_with_offset(self) -> None:
        target = _boxes((0, 0, 8, 8))
        preds = _boxes(*[(float(d), 0.0, 8.0, 8.0) for d in (0, 1, 2, 4, 8, 16)])
        scores = nwd_score(preds, target.expand_as(preds))
        self.assertTrue(bool((scores[:-1] > scores[1:]).all()))

    def test_scale_sensitivity_of_gradient_for_tiny_boxes(self) -> None:
        # A 2 px shift of an 8 px box zeroes IoU only at larger offsets; NWD still separates them.
        target = _boxes((0, 0, 8, 8))
        near = nwd_score(_boxes((9, 0, 8, 8)), target)
        far = nwd_score(_boxes((20, 0, 8, 8)), target)
        self.assertGreater(near.item(), far.item())
        self.assertGreater(far.item(), 0.0)

    def test_broadcasting_pairs_one_box_with_many(self) -> None:
        single = _boxes((0, 0, 8, 8))
        many = _boxes((0, 0, 8, 8), (3, 0, 8, 8), (0, 5, 8, 8))
        self.assertEqual(tuple(nwd_score(single, many).shape), (3,))


class TestNumericalStability(unittest.TestCase):
    def _assert_finite_grads(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.clone().requires_grad_()
        loss = NWDCoordinateLoss()(pred, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(pred.grad).all().item())

    def test_identical_boxes_have_finite_gradient(self) -> None:
        b = _boxes((10, 10, 12, 12))
        self._assert_finite_grads(b, b)

    def test_zero_dimension_boxes(self) -> None:
        self._assert_finite_grads(_boxes((5, 5, 0, 0)), _boxes((5, 5, 0, 0)))
        self._assert_finite_grads(_boxes((5, 5, 0, 4)), _boxes((6, 5, 4, 0)))

    def test_extreme_aspect_ratios(self) -> None:
        self._assert_finite_grads(_boxes((5, 5, 1e4, 1e-3)), _boxes((5, 5, 1e-3, 1e4)))

    def test_large_coordinates(self) -> None:
        self._assert_finite_grads(_boxes((1e5, 1e5, 10, 10)), _boxes((-1e5, 1e5, 10, 10)))

    def test_empty_input(self) -> None:
        empty = torch.zeros((0, 4))
        self.assertEqual(NWDCoordinateLoss(reduction="none")(empty, empty).numel(), 0)


class TestLossModules(unittest.TestCase):
    def setUp(self) -> None:
        self.pred = _boxes((10, 10, 12, 12), (30, 30, 6, 10), (50, 52, 8, 8))
        self.target = _boxes((11, 11, 12, 12), (33, 30, 6, 6), (50, 50, 8, 8))

    def test_nwd_loss_is_one_minus_score(self) -> None:
        loss = NWDCoordinateLoss(reduction="none")(self.pred, self.target)
        torch.testing.assert_close(loss, 1.0 - nwd_score(self.pred, self.target))

    def test_reductions(self) -> None:
        per_box = NWDCoordinateLoss(reduction="none")(self.pred, self.target)
        self.assertAlmostEqual(
            NWDCoordinateLoss(reduction="mean")(self.pred, self.target).item(),
            per_box.mean().item(),
            places=6,
        )
        self.assertAlmostEqual(
            NWDCoordinateLoss(reduction="sum")(self.pred, self.target).item(),
            per_box.sum().item(),
            places=6,
        )
        with self.assertRaises(ValueError):
            NWDCoordinateLoss(reduction="median")(self.pred, self.target)

    def test_hybrid_alpha_one_is_ciou(self) -> None:
        loss = HybridBoxLoss(alpha=1.0, reduction="none")(self.pred, self.target)
        ciou = bbox_iou(self.pred, self.target, xywh=True, CIoU=True).squeeze(-1)
        torch.testing.assert_close(loss, 1.0 - ciou)

    def test_hybrid_alpha_zero_is_nwd(self) -> None:
        loss = HybridBoxLoss(alpha=0.0, reduction="none")(self.pred, self.target)
        torch.testing.assert_close(
            loss, NWDCoordinateLoss(reduction="none")(self.pred, self.target)
        )

    def test_hybrid_midpoint_is_convex_combination(self) -> None:
        ciou_loss = HybridBoxLoss(alpha=1.0, reduction="none")(self.pred, self.target)
        nwd_loss = HybridBoxLoss(alpha=0.0, reduction="none")(self.pred, self.target)
        mixed = HybridBoxLoss(alpha=0.25, reduction="none")(self.pred, self.target)
        torch.testing.assert_close(mixed, 0.25 * ciou_loss + 0.75 * nwd_loss)

    def test_hybrid_xyxy_matches_xywh(self) -> None:
        def to_xyxy(b: torch.Tensor) -> torch.Tensor:
            return torch.cat((b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2), dim=1)

        xywh = HybridBoxLoss(alpha=0.5, reduction="none")(self.pred, self.target)
        xyxy = HybridBoxLoss(alpha=0.5, box_format="xyxy", reduction="none")(
            to_xyxy(self.pred), to_xyxy(self.target)
        )
        torch.testing.assert_close(xywh, xyxy, atol=1e-5, rtol=1e-5)

    def test_hybrid_broadcasts_one_box_against_many(self) -> None:
        single = self.pred[:1]
        loss = HybridBoxLoss(alpha=0.5, reduction="none")(single, self.target)
        self.assertEqual(tuple(loss.shape), (3,))
        expected = HybridBoxLoss(alpha=0.5, reduction="none")(
            single.expand_as(self.target), self.target
        )
        torch.testing.assert_close(loss, expected)

    def test_hybrid_rejects_invalid_alpha(self) -> None:
        for alpha in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                HybridBoxLoss(alpha=alpha)

    def test_hybrid_backward(self) -> None:
        pred = self.pred.clone().requires_grad_()
        HybridBoxLoss(alpha=0.5)(pred, self.target).backward()
        self.assertTrue(torch.isfinite(pred.grad).all().item())


if __name__ == "__main__":
    unittest.main()
