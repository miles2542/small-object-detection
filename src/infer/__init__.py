"""Inference pipelines including standard full-frame and SAHI tiled inference."""

from src.infer.high_res import (
    HighResConfig,
    HighResInferenceRunner,
    benchmark_resolutions,
    export_high_res_predictions,
    get_peak_memory_mb,
    reset_peak_memory,
)
from src.infer.sahi import (
    CANONICAL_CLASSES,
    CANONICAL_NAME_TO_ID,
    SahiConfig,
    SahiInferenceRunner,
    build_category_mapping,
    export_sahi_predictions,
)

__all__ = [
    "CANONICAL_CLASSES",
    "CANONICAL_NAME_TO_ID",
    "HighResConfig",
    "HighResInferenceRunner",
    "SahiConfig",
    "SahiInferenceRunner",
    "benchmark_resolutions",
    "build_category_mapping",
    "export_high_res_predictions",
    "export_sahi_predictions",
    "get_peak_memory_mb",
    "reset_peak_memory",
]
