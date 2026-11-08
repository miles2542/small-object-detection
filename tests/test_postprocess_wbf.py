"""Unit tests for Weighted Boxes Fusion (WBF) algorithm, math, and edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from src.postprocess.wbf import weighted_boxes_fusion


def test_wbf_single_box_identity() -> None:
    """Fusing a single box should return identical coordinates and score."""
    box = np.array([[100.0, 150.0, 200.0, 250.0]])
    score = np.array([0.88])
    label = np.array([3])

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        [box], [score], [label], iou_threshold=0.5
    )

    assert fused_boxes.shape == (1, 4)
    np.testing.assert_allclose(fused_boxes[0], [100.0, 150.0, 200.0, 250.0])
    assert pytest.approx(fused_scores[0], rel=1e-5) == 0.88
    assert fused_labels[0] == 3


def test_wbf_coordinate_convergence_identical_boxes() -> None:
    """Two identical boxes across two models with equal weight converge to identical box."""
    box_a = np.array([[50.0, 50.0, 70.0, 70.0]])
    score_a = np.array([0.90])
    label_a = np.array([1])

    box_b = np.array([[50.0, 50.0, 70.0, 70.0]])
    score_b = np.array([0.80])
    label_b = np.array([1])

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        [box_a, box_b],
        [score_a, score_b],
        [label_a, label_b],
        iou_threshold=0.5,
        conf_type="avg",
    )

    assert len(fused_boxes) == 1
    np.testing.assert_allclose(fused_boxes[0], [50.0, 50.0, 70.0, 70.0])
    assert pytest.approx(fused_scores[0], rel=1e-5) == 0.85
    assert fused_labels[0] == 1


def test_wbf_tiny_box_scale_invariance() -> None:
    """Micro-scale 4x4 px targets should fuse cleanly without floating-point collapse."""
    # Box 1: [10, 10, 14, 14] area = 16
    # Box 2: [11, 10, 15, 14] intersection area = 3 * 4 = 12, union = 16 + 16 - 12 = 20, IoU = 0.60
    box_1 = np.array([[10.0, 10.0, 14.0, 14.0]])
    score_1 = np.array([0.80])
    label_1 = np.array([0])

    box_2 = np.array([[11.0, 10.0, 15.0, 14.0]])
    score_2 = np.array([0.80])
    label_2 = np.array([0])

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        [box_1, box_2], [score_1, score_2], [label_1, label_2], iou_threshold=0.50
    )

    assert len(fused_boxes) == 1
    # Equal weights and scores: (10 + 11)/2 = 10.5, (14 + 15)/2 = 14.5
    np.testing.assert_allclose(fused_boxes[0], [10.5, 10.0, 14.5, 14.0])
    assert pytest.approx(fused_scores[0], rel=1e-5) == 0.80
    assert fused_labels[0] == 0


def test_wbf_disjoint_boxes_remain_separated() -> None:
    """Boxes with IoU strictly below the threshold must remain independent clusters."""
    box_1 = np.array([[10.0, 10.0, 30.0, 30.0]])
    score_1 = np.array([0.95])
    label_1 = np.array([2])

    box_2 = np.array([[100.0, 100.0, 120.0, 120.0]])
    score_2 = np.array([0.75])
    label_2 = np.array([2])

    fused_boxes, fused_scores, _ = weighted_boxes_fusion(
        [box_1, box_2], [score_1, score_2], [label_1, label_2], iou_threshold=0.50
    )

    assert len(fused_boxes) == 2
    # Result must be sorted descending by confidence
    assert pytest.approx(fused_scores[0], rel=1e-5) == 0.95
    assert pytest.approx(fused_scores[1], rel=1e-5) == 0.75
    np.testing.assert_allclose(fused_boxes[0], [10.0, 10.0, 30.0, 30.0])
    np.testing.assert_allclose(fused_boxes[1], [100.0, 100.0, 120.0, 120.0])


def test_wbf_empty_inputs() -> None:
    """Empty prediction lists or empty arrays must return valid empty arrays without error."""
    # Completely empty list
    f_b, f_s, f_l = weighted_boxes_fusion([], [], [])
    assert f_b.shape == (0, 4)
    assert f_s.shape == (0,)
    assert f_l.shape == (0,)

    # Empty model arrays
    f_b2, f_s2, f_l2 = weighted_boxes_fusion(
        [np.empty((0, 4)), np.empty((0, 4))],
        [np.empty((0,)), np.empty((0,))],
        [np.empty((0,), dtype=int), np.empty((0,), dtype=int)],
    )
    assert f_b2.shape == (0, 4)
    assert f_s2.shape == (0,)
    assert f_l2.shape == (0,)


def test_wbf_consensus_mode_avg_vs_model_scaled() -> None:
    """model_scaled mode normalizes by total model count M; avg mode normalizes by cluster size T."""
    # Model 1 detects the target with 0.80
    # Model 2 does NOT detect the target
    box_1 = np.array([[20.0, 20.0, 40.0, 40.0]])
    score_1 = np.array([0.80])
    label_1 = np.array([5])

    box_2 = np.empty((0, 4))
    score_2 = np.empty((0,))
    label_2 = np.empty((0,), dtype=int)

    # In "avg" mode, cluster size T = 1, so fused score is 0.80 / 1 = 0.80
    _, score_avg, _ = weighted_boxes_fusion(
        [box_1, box_2],
        [score_1, score_2],
        [label_1, label_2],
        conf_type="avg",
    )
    assert pytest.approx(score_avg[0], rel=1e-5) == 0.80

    # In "model_scaled" mode, total models M = 2, so fused score is 0.80 / 2 = 0.40
    _, score_scaled, _ = weighted_boxes_fusion(
        [box_1, box_2],
        [score_1, score_2],
        [label_1, label_2],
        conf_type="model_scaled",
    )
    assert pytest.approx(score_scaled[0], rel=1e-5) == 0.40


def test_wbf_custom_weights() -> None:
    """Unequal model weights should skew coordinate fusion towards the higher-weight model."""
    # Model 1: [10, 10, 20, 20], weight 3.0
    # Model 2: [20, 10, 30, 20], weight 1.0 (IoU overlap = 0.0, but let's make them overlap)
    # Model 1: [10, 10, 30, 30] (width 20)
    # Model 2: [14, 10, 34, 30] (width 20, shift x +4)
    box_1 = np.array([[10.0, 10.0, 30.0, 30.0]])
    score_1 = np.array([1.0])
    label_1 = np.array([0])

    box_2 = np.array([[14.0, 10.0, 34.0, 30.0]])
    score_2 = np.array([1.0])
    label_2 = np.array([0])

    # Weights [3.0, 1.0] -> model 1 has 3x weight of model 2
    fused_boxes, _, _ = weighted_boxes_fusion(
        [box_1, box_2],
        [score_1, score_2],
        [label_1, label_2],
        weights=[3.0, 1.0],
        iou_threshold=0.50,
    )

    # Expected x1: (3.0 * 10 + 1.0 * 14) / 4.0 = 44 / 4 = 11.0
    # Expected x2: (3.0 * 30 + 1.0 * 34) / 4.0 = 124 / 4 = 31.0
    np.testing.assert_allclose(fused_boxes[0], [11.0, 10.0, 31.0, 30.0])


def test_wbf_category_isolation() -> None:
    """Overlapping boxes of differing categories must NEVER be fused into a single detection."""
    box_a = np.array([[10.0, 10.0, 50.0, 50.0]])
    score_a = np.array([0.90])
    label_a = np.array([1])  # category 1

    box_b = np.array([[10.0, 10.0, 50.0, 50.0]])
    score_b = np.array([0.85])
    label_b = np.array([2])  # category 2

    fused_boxes, _, fused_labels = weighted_boxes_fusion(
        [box_a, box_b], [score_a, score_b], [label_a, label_b], iou_threshold=0.50
    )

    assert len(fused_boxes) == 2
    assert set(fused_labels.tolist()) == {1, 2}


def test_wbf_skip_box_threshold() -> None:
    """Predictions with confidence below skip_box_threshold must be filtered out."""
    boxes = np.array([[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]])
    scores = np.array([0.02, 0.60])
    labels = np.array([0, 0])

    fused_boxes, fused_scores, _ = weighted_boxes_fusion(
        [boxes], [scores], [labels], skip_box_threshold=0.05
    )

    assert len(fused_boxes) == 1
    assert pytest.approx(fused_scores[0], rel=1e-5) == 0.60
    np.testing.assert_allclose(fused_boxes[0], [30.0, 30.0, 40.0, 40.0])


def test_wbf_validation_errors() -> None:
    """Invalid configurations must raise descriptive ValueErrors."""
    with pytest.raises(ValueError, match="Unsupported conf_type"):
        weighted_boxes_fusion([], [], [], conf_type="median")

    with pytest.raises(ValueError, match="Length mismatch"):
        weighted_boxes_fusion([np.empty((0, 4))], [], [])

    with pytest.raises(ValueError, match="Weights length"):
        weighted_boxes_fusion(
            [np.empty((0, 4))], [np.empty((0,))], [np.empty((0,))], weights=[1.0, 2.0]
        )

    with pytest.raises(ValueError, match="must be non-negative"):
        weighted_boxes_fusion(
            [np.empty((0, 4))], [np.empty((0,))], [np.empty((0,))], weights=[-1.0]
        )
