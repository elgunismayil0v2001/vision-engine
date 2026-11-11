from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if name != "mobilenet_v3_small":
        raise ValueError(f"unsupported model {name!r}; week 1 uses mobilenet_v3_small")

    # ImageNet weights initialise the backbone; the classifier is replaced to
    # match Flowers102 (102 classes), not ImageNet's 1000.
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def load_classifier(
    *,
    model_name: str,
    num_classes: int,
    pretrained: bool,
    checkpoint: str | Path | None = None,
) -> nn.Module:
    """Build a classifier and optionally load a training checkpoint."""
    model = build_model(model_name, num_classes, pretrained)
    if checkpoint is not None:
        payload: dict[str, Any] = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: str,
    model: nn.Module,
    epoch: int,
    metrics: dict[str, float],
    extra: dict,
) -> None:
    payload = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "metrics": metrics,
        **extra,
    }
    torch.save(payload, path)
