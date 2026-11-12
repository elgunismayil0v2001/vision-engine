from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as vision_functional

VISDRONE_CLASSES = (
    "pedestrians",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning_tricycle",
    "bus",
    "motor",
)
CLASS_TO_ID = {name: index + 1 for index, name in enumerate(VISDRONE_CLASSES)}
IGNORED_LABELS = frozenset({"ignore_regions", "others"})


def load_visdrone_samples(root: str | Path) -> list[dict[str, Any]]:
    samples_path = Path(root) / "samples.json"
    if not samples_path.is_file():
        raise FileNotFoundError(
            f"VisDrone samples.json missing at {samples_path}; "
            "run python -m train.download_visdrone first"
        )
    payload: dict[str, Any] = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("VisDrone samples.json must contain a 'samples' list")
    return samples


class VisDroneDetectionDataset(Dataset):
    """Reads the Voxel51 FiftyOne export without importing FiftyOne."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        seed: int,
        horizontal_flip_probability: float = 0.0,
        max_samples: int | None = None,
        samples: list[dict[str, Any]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.seed = seed
        self.epoch = 0
        self.horizontal_flip_probability = horizontal_flip_probability
        if samples is None:
            samples = load_visdrone_samples(self.root)
        records = [sample for sample in samples if split in sample.get("tags", [])]
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be positive")
            if max_samples < len(records):
                split_seed = seed if split == "train" else seed + 1
                records = sorted(
                    random.Random(split_seed).sample(records, max_samples),
                    key=lambda sample: sample["filepath"],
                )
        if not records:
            raise ValueError(f"no VisDrone samples found for split {split!r}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self.records[index]
        image_path = Path(record["filepath"])
        if not image_path.is_absolute():
            image_path = self.root / image_path
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            width, height = image.size
            image_tensor = vision_functional.to_tensor(image)

        boxes: list[list[float]] = []
        labels: list[int] = []
        ignore_boxes: list[list[float]] = []
        ground_truth = record.get("ground_truth") or {}
        detections = ground_truth.get("detections", [])
        for detection in detections:
            label = detection.get("label")
            normalized_box = detection.get("bounding_box")
            if not isinstance(normalized_box, list) or len(normalized_box) != 4:
                continue
            x, y, box_width, box_height = (float(value) for value in normalized_box)
            x_min = min(max(x * width, 0.0), float(width))
            y_min = min(max(y * height, 0.0), float(height))
            x_max = min(max((x + box_width) * width, 0.0), float(width))
            y_max = min(max((y + box_height) * height, 0.0), float(height))
            if x_max <= x_min or y_max <= y_min:
                continue
            box = [x_min, y_min, x_max, y_max]
            if label in CLASS_TO_ID:
                boxes.append(box)
                labels.append(CLASS_TO_ID[label])
            elif label in IGNORED_LABELS:
                ignore_boxes.append(box)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        ignore_boxes_tensor = torch.tensor(ignore_boxes, dtype=torch.float32).reshape(
            -1, 4
        )
        image_id = torch.tensor(index, dtype=torch.int64)
        area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
            boxes_tensor[:, 3] - boxes_tensor[:, 1]
        )
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": image_id,
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "ignore_boxes": ignore_boxes_tensor,
        }

        flip_rng = random.Random(self.seed + self.epoch * len(self.records) + index)
        if flip_rng.random() < self.horizontal_flip_probability:
            image_tensor = image_tensor.flip(-1)
            for box_key in ("boxes", "ignore_boxes"):
                current_boxes = target[box_key]
                if current_boxes.numel() == 0:
                    continue
                old_x_min = current_boxes[:, 0].clone()
                current_boxes[:, 0] = width - current_boxes[:, 2]
                current_boxes[:, 2] = width - old_x_min
            target["area"] = (target["boxes"][:, 2] - target["boxes"][:, 0]) * (
                target["boxes"][:, 3] - target["boxes"][:, 1]
            )
        return image_tensor, target


def detection_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[tuple[torch.Tensor, ...], tuple[dict[str, torch.Tensor], ...]]:
    images, targets = zip(*batch, strict=True)
    return images, targets
