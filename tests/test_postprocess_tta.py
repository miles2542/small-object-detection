"""Unit tests for Test-Time Augmentation (TTA) coordinate transformations and inversion."""

from __future__ import annotations

import numpy as np
import pytest

from src.postprocess.tta import (
    TestTimeAugmentor,
    invert_boxes,
    transform_boxes,
    transform_image,
)


@pytest.mark.parametrize("scale", [0.85, 1.0, 1.15, 1.5, 0.5])
@pytest.mark.parametrize("hflip", [False, True])
@pytest.mark.parametrize("vflip", [False, True])
def test_tta_transform_and_inverse_symmetry(scale: float, hflip: bool, vflip: bool) -> None:
    """T^-1(T(B)) must equal B within numerical floating-point tolerances."""
    orig_shape = (800, 800)
    original_boxes = np.array(
        [
            [12.0, 15.0, 32.0, 48.0],
            [100.5, 200.25, 150.0, 250.75],
            [0.0, 0.0, 10.0, 10.0],
            [750.0, 780.0, 790.0, 800.0],
        ]
    )

    transformed = transform_boxes(
        original_boxes,
        orig_shape=orig_shape,
        scale=scale,
        hflip=hflip,
        vflip=vflip,
    )

    # Invert back to original coordinates without clipping to verify exact algebra
    inverted = invert_boxes(
        transformed,
        orig_shape=orig_shape,
        scale=scale,
        hflip=hflip,
        vflip=vflip,
        clip=False,
    )

    np.testing.assert_allclose(inverted, original_boxes, rtol=1e-5, atol=1e-5)


def test_tta_non_square_image_shape() -> None:
    """Coordinate transformation and inversion must hold for rectangular images."""
    orig_shape = (600, 1000)  # H=600, W=1000
    boxes = np.array([[50.0, 100.0, 150.0, 300.0]])

    transformed = transform_boxes(
        boxes, orig_shape=orig_shape, scale=1.2, hflip=True, vflip=True
    )
    inverted = invert_boxes(
        transformed, orig_shape=orig_shape, scale=1.2, hflip=True, vflip=True, clip=True
    )

    np.testing.assert_allclose(inverted, boxes, rtol=1e-5, atol=1e-5)


def test_tta_bounding_box_clamping() -> None:
    """Out-of-bounds predicted boxes must be clamped to [0, W] x [0, H]."""
    orig_shape = (800, 800)
    # Box extending past borders
    raw_transformed_boxes = np.array([[-20.0, -10.0, 850.0, 900.0]])

    clamped = invert_boxes(
        raw_transformed_boxes,
        orig_shape=orig_shape,
        scale=1.0,
        hflip=False,
        vflip=False,
        clip=True,
    )

    np.testing.assert_allclose(clamped[0], [0.0, 0.0, 800.0, 800.0])


def test_tta_image_transform_shapes() -> None:
    """transform_image must correctly scale and orient image arrays."""
    dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)
    dummy_img[0:10, 0:10, :] = 255  # marker at top-left

    # Horizontal flip should move marker to top-right
    h_flipped = transform_image(dummy_img, hflip=True)
    assert h_flipped.shape == (100, 200, 3)
    assert np.all(h_flipped[0:10, 190:200, :] == 255)

    # Scaling
    scaled = transform_image(dummy_img, scale=0.5)
    assert scaled.shape == (50, 100, 3)


def test_tta_augmentor_forward_execution() -> None:
    """TestTimeAugmentor forward pass should generate augmented views and fuse predictions."""
    dummy_img = np.zeros((800, 800, 3), dtype=np.uint8)
    target_box = np.array([[100.0, 100.0, 120.0, 120.0]])

    # Mock model that consistently returns target_box transformed to the view coordinates
    def mock_model(aug_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _h, w = aug_img.shape[:2]
        # Box scaled to current image dimensions
        scale = w / 800.0
        b = target_box * scale
        s = np.array([0.90])
        lbl = np.array([0])
        return b, s, lbl

    augmentor = TestTimeAugmentor(scales=[1.0, 1.15], hflip=False, vflip=False)
    fused_boxes, fused_scores, fused_labels = augmentor.forward(mock_model, dummy_img)

    assert len(fused_boxes) == 1
    np.testing.assert_allclose(fused_boxes[0], [100.0, 100.0, 120.0, 120.0], atol=1e-3)
    assert pytest.approx(fused_scores[0], rel=1e-3) == 0.90
    assert fused_labels[0] == 0
