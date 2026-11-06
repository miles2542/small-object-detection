"""Bridge the hybrid CIoU/NWD box loss into the Ultralytics detection training loop.

Ultralytics builds its criterion lazily inside ``DetectionModel.init_criterion``. Instead of
storing anything on the model instance (which would end up in saved checkpoints), the trainer
returned by :func:`make_nwd_trainer` swaps in a criterion factory at class level for each
training process, so single- and multi-GPU runs behave the same.

The regression loss of both YOLOv8 (DFL head) and YOLO26 (end-to-end, ``reg_max=1``) goes through
``BboxLoss``; only its IoU term is replaced, the DFL / L1 term is left untouched.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import BboxLoss

from src.losses.nwd import DEFAULT_HYBRID_ALPHA, DEFAULT_NWD_CONSTANT, HybridBoxLoss

# Captured at import so the class-level patch installed during training can always be undone.
_ORIGINAL_INIT_CRITERION = DetectionModel.init_criterion


class NWDBboxLoss(BboxLoss):
    """``BboxLoss`` whose box term is ``alpha * CIoU + (1 - alpha) * NWD`` in pixel units."""

    def __init__(
        self,
        reg_max: int = 16,
        alpha: float = DEFAULT_HYBRID_ALPHA,
        constant: float = DEFAULT_NWD_CONSTANT,
    ) -> None:
        super().__init__(reg_max)
        self.box_loss = HybridBoxLoss(
            alpha=alpha, constant=constant, box_format="xyxy", reduction="none"
        )

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The parent also computes its CIoU term, which is discarded; reusing it keeps the DFL / L1
        # branch identical to the installed Ultralytics version.
        _, loss_dfl = super().forward(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride,
        )

        fg = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[fg].sum(-1)
        # Boxes arrive in feature-grid units; NWD's constant is defined in pixels.
        # Mixed precision hands over half-precision boxes; pixel-scaled coordinates need float32.
        scale = stride[fg[1]].float()
        box = self.box_loss(pred_bboxes[fg].float() * scale, target_bboxes[fg].float() * scale)
        loss_box = (box * weight).sum() / target_scores_sum
        return loss_box, loss_dfl


def _detection_losses(criterion: Any) -> Iterator[Any]:
    """Yield every ``v8DetectionLoss`` inside a criterion (one for v8, two for end-to-end)."""
    if isinstance(getattr(criterion, "bbox_loss", None), BboxLoss):
        yield criterion
    for name in ("one2many", "one2one"):
        child = getattr(criterion, name, None)
        if child is not None:
            yield from _detection_losses(child)


def apply_nwd_loss(
    criterion: Any,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    constant: float = DEFAULT_NWD_CONSTANT,
) -> Any:
    """Replace the box loss of an Ultralytics detection criterion in place and return it."""
    patched = 0
    for loss in _detection_losses(criterion):
        loss.bbox_loss = NWDBboxLoss(loss.reg_max, alpha, constant).to(loss.device)
        patched += 1
    if patched == 0:
        raise TypeError(f"No BboxLoss found in criterion of type {type(criterion).__name__}")
    return criterion


def make_nwd_trainer(
    nwd_ratio: float = DEFAULT_HYBRID_ALPHA,
    nwd_constant: float = DEFAULT_NWD_CONSTANT,
) -> type[DetectionTrainer]:
    """Return a ``DetectionTrainer`` subclass that trains with the hybrid CIoU/NWD box loss.

    ``nwd_ratio`` is the CIoU weight alpha (``0`` is pure NWD, ``1`` is pure CIoU) and
    ``nwd_constant`` is the NWD normalization constant ``C`` in pixels.
    """
    HybridBoxLoss(alpha=nwd_ratio, constant=nwd_constant)  # validate eagerly

    class NWDDetectionTrainer(DetectionTrainer):
        alpha = nwd_ratio
        constant = nwd_constant

        def set_model_attributes(self) -> None:
            super().set_model_attributes()
            alpha, constant = self.alpha, self.constant

            def init_criterion(model: DetectionModel) -> Any:
                return apply_nwd_loss(_ORIGINAL_INIT_CRITERION(model), alpha, constant)

            DetectionModel.init_criterion = init_criterion  # type: ignore[method-assign]

        def train(self, *args: Any, **kwargs: Any) -> Any:
            # The patch is class-wide, so it must not outlive this run.
            try:
                return super().train(*args, **kwargs)
            finally:
                DetectionModel.init_criterion = _ORIGINAL_INIT_CRITERION  # type: ignore[method-assign]

    return NWDDetectionTrainer
