"""Build and verify YOLO detection models from architecture configs (e.g. the P2 variants)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops


@dataclass
class ModelReport:
    """Static facts about a detection model and the feature maps it feeds its head."""

    config: str
    imgsz: int
    num_classes: int
    parameters: int
    gflops: float
    strides: list[int]
    feature_shapes: list[tuple[int, ...]]
    end2end: bool
    receptive_fields: list[int] = field(default_factory=list)

    @property
    def head_count(self) -> int:
        return len(self.strides)


def build_model(
    config_path: str | Path,
    num_classes: int = 8,
    weights: str | Path | None = None,
) -> YOLO:
    """Instantiate a detection model from a YAML config and verify its head geometry.

    ``weights`` (optional) transfers every tensor whose name and shape match, which covers the
    unchanged backbone of a P2 variant; the rest keeps its fresh initialization.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found at: {config_path}")

    model = YOLO(str(config_path))
    if model.model.yaml["nc"] != num_classes:
        model.model = DetectionModel(model.model.yaml, nc=num_classes, verbose=False)
    if weights is not None:
        model.load(str(weights))

    verify_model(model, imgsz=800)
    return model


def _capture_head_inputs(model: YOLO, x: torch.Tensor) -> list[torch.Tensor]:
    """Run ``x`` through the network and return the feature maps entering the detect head."""
    captured: list[torch.Tensor] = []
    head = model.model.model[-1]
    handle = head.register_forward_pre_hook(lambda _m, args: captured.extend(args[0]))
    try:
        model.model(x)
    finally:
        handle.remove()
    return captured


def verify_model(model: YOLO, imgsz: int = 800) -> ModelReport:
    """Forward a dummy image and check that feature maps match the head strides."""
    net = model.model.eval()
    with torch.no_grad():
        feats = _capture_head_inputs(model, torch.zeros((1, 3, imgsz, imgsz)))

    head = net.model[-1]
    strides = [int(s) for s in head.stride.tolist()]
    shapes = [tuple(f.shape) for f in feats]
    if len(shapes) != len(strides):
        raise ValueError(f"Head has {len(strides)} strides but received {len(shapes)} feature maps")
    for stride, shape in zip(strides, shapes, strict=True):
        expected = imgsz // stride
        if shape[-2:] != (expected, expected):
            raise ValueError(
                f"Stride {stride}: expected {expected}x{expected} feature map, got {shape[-2:]}"
            )

    return ModelReport(
        config=str(getattr(net, "yaml_file", "")),
        imgsz=imgsz,
        num_classes=int(head.nc),
        parameters=sum(p.numel() for p in net.parameters()),
        gflops=float(get_flops(net, imgsz)),
        strides=strides,
        feature_shapes=shapes,
        end2end=getattr(head, "one2one_cv2", None) is not None,
    )


def measure_receptive_fields(model: YOLO, imgsz: int = 800) -> list[int]:
    """Empirical receptive field (pixels, per axis) of the center cell at each head scale.

    Measured as the extent of non-zero input gradient, so global modules such as attention
    show up as the full image.
    """
    net = model.model.eval()
    x = torch.zeros((1, 3, imgsz, imgsz), requires_grad=True)
    with torch.enable_grad():
        feats = _capture_head_inputs(model, x)
        fields: list[int] = []
        for feat in feats:
            x.grad = None
            h, w = feat.shape[-2:]
            feat[0, :, h // 2, w // 2].sum().backward(retain_graph=True)
            active = (x.grad.abs().sum(1)[0] > 0).nonzero()
            span = active.max(0).values - active.min(0).values + 1
            fields.append(int(span.max()))
    net.zero_grad(set_to_none=True)
    return fields
