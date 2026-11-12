from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from train.detection_metrics import compute_map
from train.detector import _resolve_device, build_detector
from train.detector_config import DetectorConfig
from train.visdrone_data import VisDroneDetectionDataset


def _sample(filepath: str, split: str) -> dict[str, object]:
    return {
        "filepath": filepath,
        "tags": [split],
        "ground_truth": {
            "detections": [
                {"label": "pedestrians", "bounding_box": [0.1, 0.2, 0.3, 0.4]},
                {"label": "ignore_regions", "bounding_box": [0.5, 0.2, 0.2, 0.3]},
                {"label": "others", "bounding_box": [0.0, 0.0, 0.1, 0.1]},
            ]
        },
    }


def test_visdrone_dataset_maps_boxes_labels_and_ignore_regions(tmp_path: Path) -> None:
    image_path = tmp_path / "data" / "frame.jpg"
    image_path.parent.mkdir()
    Image.new("RGB", (100, 50), color=(10, 20, 30)).save(image_path)
    sample = _sample("data/frame.jpg", "train")

    dataset = VisDroneDetectionDataset(
        tmp_path,
        "train",
        seed=5,
        horizontal_flip_probability=0.0,
        samples=[sample],
    )
    image, target = dataset[0]

    assert image.shape == (3, 50, 100)
    assert target["boxes"].tolist() == [[10.0, 10.0, 40.0, 30.0]]
    assert target["labels"].tolist() == [1]
    assert target["ignore_boxes"].tolist() == [
        [50.0, 10.0, 70.0, 25.0],
        [0.0, 0.0, 10.0, 5.0],
    ]


def test_visdrone_dataset_horizontal_flip_updates_boxes(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 50)).save(image_path)
    dataset = VisDroneDetectionDataset(
        tmp_path,
        "train",
        seed=5,
        horizontal_flip_probability=1.0,
        samples=[_sample("frame.jpg", "train")],
    )

    _, target = dataset[0]

    assert target["boxes"].tolist() == [[60.0, 10.0, 90.0, 30.0]]
    assert target["ignore_boxes"][0].tolist() == [30.0, 10.0, 50.0, 25.0]


def _target(box: list[float], *, ignore: list[list[float]] | None = None):
    return {
        "boxes": torch.tensor([box], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "ignore_boxes": torch.tensor(ignore or [], dtype=torch.float32).reshape(-1, 4),
    }


def _prediction(
    boxes: list[list[float]], scores: list[float], labels: list[int] | None = None
):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels or [1] * len(scores), dtype=torch.int64),
    }


def test_detection_map_is_one_for_perfect_single_class_prediction() -> None:
    metrics = compute_map(
        [_prediction([[0, 0, 10, 10]], [0.9])],
        [_target([0, 0, 10, 10])],
        iou_thresholds=[0.5, 0.75],
    )

    assert metrics.map_50_95 == pytest.approx(1.0)
    assert metrics.map_50 == pytest.approx(1.0)
    assert metrics.per_class_map_50["pedestrians"] == pytest.approx(1.0)


def test_detection_map_does_not_count_predictions_inside_ignore_boxes() -> None:
    target = _target([0, 0, 10, 10], ignore=[[20, 20, 30, 30]])
    clean = _prediction([[0, 0, 10, 10]], [0.8])
    with_ignored_detection = _prediction(
        [[20, 20, 30, 30], [0, 0, 10, 10]], [0.99, 0.8]
    )

    clean_metrics = compute_map([clean], [target], iou_thresholds=[0.5])
    ignored_metrics = compute_map(
        [with_ignored_detection], [target], iou_thresholds=[0.5]
    )

    assert ignored_metrics.map_50 == pytest.approx(clean_metrics.map_50)


def test_detector_replaces_coco_predictor_with_visdrone_classes() -> None:
    cfg = DetectorConfig.from_yaml("train/configs/visdrone.yaml")

    model = build_detector(cfg, use_coco_weights=False)

    assert model.roi_heads.box_predictor.cls_score.out_features == 11
    assert model.roi_heads.box_predictor.bbox_pred.out_features == 44


def test_detector_can_force_cpu_fallback() -> None:
    assert _resolve_device("cpu") == torch.device("cpu")
