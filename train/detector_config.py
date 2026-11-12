from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DetectorConfig:
    seed: int
    dataset_root: str
    dataset_repo: str
    dataset_revision: str
    model_name: str
    num_classes: int
    use_coco_weights: bool
    epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    weight_decay: float
    lr_step_size: int
    lr_gamma: float
    trainable_backbone_layers: int
    horizontal_flip_probability: float
    score_threshold: float
    max_detections_per_image: int
    checkpoint_every_steps: int
    num_workers: int
    torch_threads: int
    download_workers: int
    output_dir: str
    max_train_samples: int | None
    max_val_samples: int | None
    iou_thresholds: list[float]
    recall_points: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> DetectorConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.model_name != "fasterrcnn_mobilenet_v3_large_320_fpn":
            raise ValueError(f"unsupported detector model: {self.model_name}")
        if self.num_classes != 11:
            raise ValueError("VisDrone uses 10 foreground classes plus background")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay nonnegative"
            )
        if self.momentum < 0 or self.lr_step_size < 1 or not 0 < self.lr_gamma <= 1:
            raise ValueError("invalid optimizer or scheduler settings")
        if not 0 <= self.trainable_backbone_layers <= 6:
            raise ValueError("trainable_backbone_layers must be between 0 and 6")
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if not 0 <= self.score_threshold < 1:
            raise ValueError("score_threshold must be in [0, 1)")
        if self.max_detections_per_image < 1 or self.checkpoint_every_steps < 1:
            raise ValueError("detection and checkpoint intervals must be positive")
        if self.num_workers < 0 or self.torch_threads < 1 or self.download_workers < 1:
            raise ValueError("worker counts are invalid")
        for cap in (self.max_train_samples, self.max_val_samples):
            if cap is not None and cap < 1:
                raise ValueError("sample caps must be positive or null")
        if not self.iou_thresholds or any(
            not 0 < threshold <= 1 for threshold in self.iou_thresholds
        ):
            raise ValueError("iou_thresholds must contain values in (0, 1]")
        if sorted(self.iou_thresholds) != self.iou_thresholds:
            raise ValueError("iou_thresholds must be sorted")
        if self.recall_points < 2:
            raise ValueError("recall_points must be at least 2")
