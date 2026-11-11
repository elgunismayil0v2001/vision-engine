from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainConfig:
    """All knobs for a training run. Numbers live here, not in train.py."""

    seed: int
    dataset: str
    data_root: str
    model_name: str
    pretrained: bool
    num_classes: int
    image_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    output_dir: str
    # Cap samples per split for smoke runs. None = use the full split.
    max_train_samples: int | None
    max_val_samples: int | None

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
