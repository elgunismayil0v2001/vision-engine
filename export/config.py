from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExportConfig:
    seed: int
    model_name: str
    num_classes: int
    image_size: int
    opset: int
    onnx_path: str
    checkpoint: str | None
    pretrained: bool

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExportConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        return cls(**raw)
