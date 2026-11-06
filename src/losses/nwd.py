"""Normalized Gaussian Wasserstein Distance (NWD) loss for tiny-object box regression.

A horizontal box ``(cx, cy, w, h)`` is modeled as a 2D Gaussian with mean ``(cx, cy)`` and
covariance ``diag(w^2 / 4, h^2 / 4)``. For two such Gaussians the squared 2-Wasserstein distance
has the closed form::

    W2^2 = (cx_a - cx_b)^2 + (cy_a - cy_b)^2 + ((w_a - w_b)^2 + (h_a - h_b)^2) / 4

and the normalized similarity is ``NWD = exp(-sqrt(W2^2) / C)``. Unlike IoU, NWD stays smooth and
non-zero for boxes that do not overlap, which keeps gradients alive for sub-16 px targets.

All functions are elementwise over broadcastable ``(..., 4)`` tensors and expect box sizes in the
same unit as ``constant`` (pixels for the default ``C``).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from ultralytics.utils.metrics import bbox_iou

BoxFormat = Literal["xywh", "xyxy"]

DEFAULT_NWD_CONSTANT = 12.8
DEFAULT_HYBRID_ALPHA = 0.5
_EPS = 1e-7


def _to_xywh(boxes: Tensor, box_format: BoxFormat) -> Tensor:
    """Return boxes as ``[cx, cy, w, h]``."""
    if box_format == "xywh":
        return boxes
    if box_format == "xyxy":
        x1, y1, x2, y2 = boxes.unbind(-1)
        return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)
    raise ValueError(f"Unsupported box_format: {box_format!r} (expected 'xywh' or 'xyxy')")


def wasserstein_distance_2(
    boxes1: Tensor, boxes2: Tensor, box_format: BoxFormat = "xywh"
) -> Tensor:
    """Squared 2-Wasserstein distance between the Gaussians of two box sets, shape ``(...)``."""
    a = _to_xywh(boxes1, box_format)
    b = _to_xywh(boxes2, box_format)
    center_sq = (a[..., :2] - b[..., :2]).pow(2).sum(-1)
    size_sq = (a[..., 2:] - b[..., 2:]).pow(2).sum(-1) / 4
    return (center_sq + size_sq).clamp(min=0)


def nwd_score(
    boxes1: Tensor,
    boxes2: Tensor,
    constant: float = DEFAULT_NWD_CONSTANT,
    box_format: BoxFormat = "xywh",
) -> Tensor:
    """Normalized Wasserstein similarity in ``(0, 1]``, shape ``(...)``."""
    if constant <= 0:
        raise ValueError(f"constant must be positive, got {constant}")
    w2 = wasserstein_distance_2(boxes1, boxes2, box_format)
    # sqrt has an infinite gradient at 0, so coincident boxes take a constant-zero branch.
    distance = torch.where(w2 > 0, torch.sqrt(w2.clamp(min=_EPS)), torch.zeros_like(w2))
    return torch.exp(-distance / constant)


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction!r} (expected 'none', 'mean' or 'sum')")


class NWDCoordinateLoss(nn.Module):
    """``1 - NWD`` box loss."""

    def __init__(
        self,
        constant: float = DEFAULT_NWD_CONSTANT,
        box_format: BoxFormat = "xywh",
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.constant = constant
        self.box_format = box_format
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = 1.0 - nwd_score(pred, target, self.constant, self.box_format)
        return _reduce(loss, self.reduction)


class HybridBoxLoss(nn.Module):
    """``alpha * (1 - CIoU) + (1 - alpha) * (1 - NWD)``.

    ``alpha=1`` is pure CIoU and ``alpha=0`` is pure NWD.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_HYBRID_ALPHA,
        constant: float = DEFAULT_NWD_CONSTANT,
        box_format: BoxFormat = "xywh",
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.constant = constant
        self.box_format = box_format
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred, target = torch.broadcast_tensors(pred, target)
        loss = pred.new_zeros(pred.shape[:-1])
        if self.alpha > 0.0:
            iou = bbox_iou(pred, target, xywh=self.box_format == "xywh", CIoU=True)
            loss = loss + self.alpha * (1.0 - iou.reshape(loss.shape))
        if self.alpha < 1.0:
            nwd = nwd_score(pred, target, self.constant, self.box_format)
            loss = loss + (1.0 - self.alpha) * (1.0 - nwd)
        return _reduce(loss, self.reduction)
