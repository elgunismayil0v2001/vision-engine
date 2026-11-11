from __future__ import annotations

import torch


def pick_device() -> torch.device:
    """Prefer Apple MPS, then CUDA, else CPU. CUDA is never required."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
