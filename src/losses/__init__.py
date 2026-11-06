"""Loss functions for tiny-object detection."""

from src.losses.nwd import (
    HybridBoxLoss,
    NWDCoordinateLoss,
    nwd_score,
    wasserstein_distance_2,
)

__all__ = ["HybridBoxLoss", "NWDCoordinateLoss", "nwd_score", "wasserstein_distance_2"]
