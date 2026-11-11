from __future__ import annotations

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
