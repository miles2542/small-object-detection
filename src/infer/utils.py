"""Shared utilities and constants for inference modules."""

from __future__ import annotations

from typing import Any

import psutil
import torch

CANONICAL_CLASSES: dict[int, str] = {
    0: "airplane",
    1: "bridge",
    2: "storage-tank",
    3: "ship",
    4: "swimming-pool",
    5: "vehicle",
    6: "person",
    7: "wind-mill",
}

CANONICAL_NAME_TO_ID: dict[str, int] = {v: k for k, v in CANONICAL_CLASSES.items()}


def parse_cuda_device_index(device: str = "cpu") -> int | None:
    """Resolve CUDA device index from device strings such as '0', 'cuda:0', 'cuda:1'.

    Returns None if the device is not CUDA or CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return None

    dev_str = str(device).strip().lower()
    if dev_str == "cpu":
        return None

    if dev_str.isdigit():
        return int(dev_str)

    if dev_str.startswith("cuda"):
        try:
            return torch.device(dev_str).index or 0
        except (ValueError, RuntimeError):
            return torch.cuda.current_device()

    return None


def get_peak_memory_mb(device: str = "cpu") -> float:
    """Retrieve peak memory consumption in megabytes (GPU VRAM or CPU RAM)."""
    dev_idx = parse_cuda_device_index(device)
    if dev_idx is not None:
        return float(torch.cuda.max_memory_allocated(dev_idx) / (1024 * 1024))
    process = psutil.Process()
    return float(process.memory_info().rss / (1024 * 1024))


def reset_peak_memory(device: str = "cpu") -> None:
    """Reset peak memory tracking counters if supported on the device."""
    dev_idx = parse_cuda_device_index(device)
    if dev_idx is not None:
        torch.cuda.reset_peak_memory_stats(dev_idx)


def build_category_mapping(
    model_names: dict[int, str] | list[str],
    target_categories: list[dict[str, Any]] | dict[int, str] | None = None,
    smoke: bool = False,
) -> dict[int, int]:
    """Map model prediction indices to target dataset category IDs.

    Ensures exact name-based matching, never index assumptions.
    """
    names = dict(enumerate(model_names)) if isinstance(model_names, list) else model_names

    if target_categories is None:
        target = CANONICAL_NAME_TO_ID
    elif isinstance(target_categories, list):
        target = {cat["name"]: cat["id"] for cat in target_categories}
    else:
        target = {name: cid for cid, name in target_categories.items()}

    if (
        len(target) == 0
        or len(set(target.values())) != len(target)
        or len(set(names.values())) != len(names)
        or any(not isinstance(idx, int) or idx < 0 for idx in names)
    ):
        raise ValueError("Category names and IDs must be non-empty, unique, and non-negative")

    if not smoke and set(names.values()) != set(target):
        raise ValueError(
            "Model vocabulary differs from target ground truth; use smoke=True for partial tests"
        )

    mapping = {idx: target[name] for idx, name in names.items() if name in target}
    if not mapping:
        raise ValueError("No matching category names found between model and target")
    return mapping
