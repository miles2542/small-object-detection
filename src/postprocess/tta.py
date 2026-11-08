"""Test-Time Augmentation (TTA) orchestrator with exact coordinate inversion."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.postprocess.wbf import weighted_boxes_fusion


def transform_image(
    image: np.ndarray,
    scale: float = 1.0,
    hflip: bool = False,
    vflip: bool = False,
) -> np.ndarray:
    """Apply spatial test-time transformations to an image array."""
    out = image.copy()
    if hflip:
        out = np.ascontiguousarray(np.fliplr(out))
    if vflip:
        out = np.ascontiguousarray(np.flipud(out))
    if abs(scale - 1.0) > 1e-6:
        h, w = out.shape[:2]
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return out


def transform_boxes(
    boxes: np.ndarray | Sequence[Sequence[float]],
    orig_shape: tuple[int, int],
    scale: float = 1.0,
    hflip: bool = False,
    vflip: bool = False,
) -> np.ndarray:
    """Forward-transform bounding boxes [x1, y1, x2, y2] to match TTA image transforms."""
    b = np.asarray(boxes, dtype=float)
    if b.size == 0:
        return np.empty((0, 4), dtype=float)

    out = b.copy()
    orig_h, orig_w = orig_shape

    if abs(scale - 1.0) > 1e-6:
        out *= scale

    curr_w = orig_w * scale
    curr_h = orig_h * scale

    if hflip:
        x1 = curr_w - out[:, 2]
        x2 = curr_w - out[:, 0]
        out[:, 0] = x1
        out[:, 2] = x2

    if vflip:
        y1 = curr_h - out[:, 3]
        y2 = curr_h - out[:, 1]
        out[:, 1] = y1
        out[:, 3] = y2

    return out


def invert_boxes(
    boxes: np.ndarray | Sequence[Sequence[float]],
    orig_shape: tuple[int, int],
    scale: float = 1.0,
    hflip: bool = False,
    vflip: bool = False,
    clip: bool = True,
) -> np.ndarray:
    """Invert transformed bounding boxes back to canonical unaugmented image coordinates."""
    b = np.asarray(boxes, dtype=float)
    if b.size == 0:
        return np.empty((0, 4), dtype=float)

    out = b.copy()
    orig_h, orig_w = orig_shape

    curr_w = orig_w * scale
    curr_h = orig_h * scale

    if vflip:
        y1 = curr_h - out[:, 3]
        y2 = curr_h - out[:, 1]
        out[:, 1] = y1
        out[:, 3] = y2

    if hflip:
        x1 = curr_w - out[:, 2]
        x2 = curr_w - out[:, 0]
        out[:, 0] = x1
        out[:, 2] = x2

    if abs(scale - 1.0) > 1e-6:
        out /= scale

    if clip:
        out[:, 0] = np.clip(out[:, 0], 0.0, float(orig_w))
        out[:, 1] = np.clip(out[:, 1], 0.0, float(orig_h))
        out[:, 2] = np.clip(out[:, 2], 0.0, float(orig_w))
        out[:, 3] = np.clip(out[:, 3], 0.0, float(orig_h))

    return out


class TestTimeAugmentor:
    """Orchestrates test-time augmentation inference with coordinate inversion and WBF fusion."""

    __test__ = False

    def __init__(
        self,
        scales: Sequence[float] = (1.0,),
        hflip: bool = True,
        vflip: bool = False,
        iou_threshold: float = 0.50,
        skip_box_threshold: float = 0.05,
        conf_type: str = "avg",
    ) -> None:
        if not scales:
            raise ValueError("scales sequence must not be empty")
        if any(s <= 0.0 for s in scales):
            raise ValueError("All scale factors must be positive")

        self.scales = list(scales)
        self.hflip = hflip
        self.vflip = vflip
        self.iou_threshold = iou_threshold
        self.skip_box_threshold = skip_box_threshold
        self.conf_type = conf_type

    def _get_augmentation_configurations(self) -> list[tuple[float, bool, bool]]:
        """Generate list of (scale, hflip, vflip) configurations."""
        configs: list[tuple[float, bool, bool]] = []
        for s in self.scales:
            configs.append((s, False, False))
            if self.hflip:
                configs.append((s, True, False))
            if self.vflip:
                configs.append((s, False, True))
            if self.hflip and self.vflip:
                configs.append((s, True, True))
        return configs

    def forward(
        self,
        model: Any,
        image: np.ndarray | Path | str | Image.Image,
        **inference_kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute multi-view augmented inference and fuse predictions via WBF.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            fused_boxes (N, 4), fused_scores (N,), fused_labels (N,) in canonical coordinates.
        """
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.is_file():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            img_arr = cv2.imread(str(img_path))
            if img_arr is None:
                raise ValueError(f"Failed to read image at: {img_path}")
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img_arr = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            img_arr = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        orig_h, orig_w = img_arr.shape[:2]
        configs = self._get_augmentation_configurations()

        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for s, hf, vf in configs:
            aug_img = transform_image(img_arr, scale=s, hflip=hf, vflip=vf)

            # Support both Ultralytics YOLO models and generic callables
            if hasattr(model, "predict"):
                preds = model.predict(aug_img, verbose=False, **inference_kwargs)
                if isinstance(preds, list) and len(preds) > 0 and hasattr(preds[0], "boxes"):
                    boxes_obj = preds[0].boxes
                    view_boxes = boxes_obj.xyxy.cpu().numpy()
                    view_scores = boxes_obj.conf.cpu().numpy()
                    view_labels = boxes_obj.cls.cpu().numpy().astype(int)
                else:
                    view_boxes = np.empty((0, 4), dtype=float)
                    view_scores = np.empty((0,), dtype=float)
                    view_labels = np.empty((0,), dtype=int)
            elif callable(model):
                out = model(aug_img, **inference_kwargs)
                if isinstance(out, tuple) and len(out) == 3:
                    view_boxes = np.asarray(out[0], dtype=float)
                    view_scores = np.asarray(out[1], dtype=float)
                    view_labels = np.asarray(out[2], dtype=int)
                elif hasattr(out, "boxes"):
                    boxes_obj = out.boxes
                    view_boxes = boxes_obj.xyxy.cpu().numpy()
                    view_scores = boxes_obj.conf.cpu().numpy()
                    view_labels = boxes_obj.cls.cpu().numpy().astype(int)
                else:
                    raise ValueError(f"Unexpected output structure from model callable: {type(out)}")
            else:
                raise TypeError(f"Model object of type {type(model)} is neither callable nor has .predict()")

            # Invert coordinates to canonical original image dimensions
            inv_boxes = invert_boxes(
                view_boxes,
                orig_shape=(orig_h, orig_w),
                scale=s,
                hflip=hf,
                vflip=vf,
                clip=True,
            )

            all_boxes.append(inv_boxes)
            all_scores.append(view_scores)
            all_labels.append(view_labels)

        return weighted_boxes_fusion(
            all_boxes,
            all_scores,
            all_labels,
            iou_threshold=self.iou_threshold,
            skip_box_threshold=self.skip_box_threshold,
            conf_type=self.conf_type,
        )
