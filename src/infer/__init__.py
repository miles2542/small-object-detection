"""Inference pipelines including standard full-frame and SAHI tiled inference."""

from src.infer.high_res import (
    HighResConfig,
    HighResInferenceRunner,
    benchmark_resolutions,
    export_high_res_predictions,
)
from src.infer.sahi import (
    SahiConfig,
    SahiInferenceRunner,
    export_sahi_predictions,
)
from src.infer.utils import (
    CANONICAL_CLASSES,
    CANONICAL_NAME_TO_ID,
    build_category_mapping,
    get_peak_memory_mb,
    parse_cuda_device_index,
    reset_peak_memory,
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
    "parse_cuda_device_index",
    "reset_peak_memory",
]
