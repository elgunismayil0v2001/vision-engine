from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    n = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * targets.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        n += targets.size(0)
    if n == 0:
        return {"val_loss": 0.0, "val_acc": 0.0, "n": 0.0}
    return {
        "val_loss": total_loss / n,
        "val_acc": correct / n,
        "n": float(n),
    }
