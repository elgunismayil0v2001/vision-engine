from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BenchmarkConfig:
    """Settings for a reproducible CPU comparison of classifier variants."""

    seed: int
    train_config: str
    checkpoint: str
    output_dir: str
    warmup_iterations: int
    benchmark_iterations: int
    num_threads: int
    pruning_amount: float
    quantization_backend: str
    calibration_batches: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        return cls(**raw)

    def validate(self) -> None:
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if self.benchmark_iterations < 2:
            raise ValueError("benchmark_iterations must be at least 2")
        if self.num_threads < 1:
            raise ValueError("num_threads must be positive")
        if not 0.0 <= self.pruning_amount < 1.0:
            raise ValueError("pruning_amount must be in [0, 1)")
        if self.calibration_batches < 1:
            raise ValueError("calibration_batches must be positive")
