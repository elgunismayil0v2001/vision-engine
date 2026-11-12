from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from train.visdrone_data import VISDRONE_CLASSES


@dataclass(frozen=True)
class DetectionMetrics:
    map_50_95: float
    map_50: float
    per_class_map_50_95: dict[str, float]
    per_class_map_50: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "map_50_95": self.map_50_95,
            "map_50": self.map_50,
            "per_class_map_50_95": self.per_class_map_50_95,
            "per_class_map_50": self.per_class_map_50,
        }


def _as_numpy_boxes(value: torch.Tensor) -> np.ndarray:
    return value.detach().to(device="cpu", dtype=torch.float32).numpy().reshape(-1, 4)


def _iou_with_box(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    intersection_wh = np.maximum(right_bottom - left_top, 0.0)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    box_area = max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)
    boxes_wh = np.maximum(boxes[:, 2:] - boxes[:, :2], 0.0)
    boxes_area = boxes_wh[:, 0] * boxes_wh[:, 1]
    union = box_area + boxes_area - intersection
    return np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )


def _average_precision(
    true_positives: list[int],
    false_positives: list[int],
    ground_truth_count: int,
    recall_points: int,
) -> float:
    if ground_truth_count == 0:
        return float("nan")
    if not true_positives:
        return 0.0
    true_positive_sum = np.cumsum(np.asarray(true_positives, dtype=np.float64))
    false_positive_sum = np.cumsum(np.asarray(false_positives, dtype=np.float64))
    recall = true_positive_sum / ground_truth_count
    precision = true_positive_sum / np.maximum(
        true_positive_sum + false_positive_sum, 1e-12
    )
    recall_curve = np.concatenate(([0.0], recall, [1.0]))
    precision_curve = np.concatenate(([0.0], precision, [0.0]))
    for index in range(precision_curve.size - 2, -1, -1):
        precision_curve[index] = max(precision_curve[index], precision_curve[index + 1])
    sample_recalls = np.linspace(0.0, 1.0, recall_points)
    sampled_precision = [
        precision_curve[np.searchsorted(recall_curve, sample_recall, side="left")]
        for sample_recall in sample_recalls
    ]
    return float(np.mean(sampled_precision))


def compute_map(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    *,
    iou_thresholds: list[float],
    recall_points: int = 101,
    max_detections_per_image: int = 100,
) -> DetectionMetrics:
    """Compute class-averaged 101-point AP over the supplied IoU thresholds."""
    if len(predictions) != len(targets):
        raise ValueError("prediction and target image counts differ")
    if not predictions:
        raise ValueError("mAP requires at least one image")
    if not iou_thresholds or any(
        not 0 < threshold <= 1 for threshold in iou_thresholds
    ):
        raise ValueError("IoU thresholds must be in (0, 1]")
    if recall_points < 2 or max_detections_per_image < 1:
        raise ValueError("recall_points and max_detections_per_image must be positive")

    gt_by_class: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    ignore_by_image: list[np.ndarray] = []
    prediction_by_class: dict[int, list[tuple[float, int, np.ndarray]]] = defaultdict(
        list
    )
    gt_counts = {class_id: 0 for class_id in range(1, len(VISDRONE_CLASSES) + 1)}

    for image_index, target in enumerate(targets):
        boxes = _as_numpy_boxes(target["boxes"])
        labels = target["labels"].detach().to(device="cpu").numpy().reshape(-1)
        for class_id in gt_counts:
            class_boxes = boxes[labels == class_id]
            gt_by_class[class_id][image_index] = class_boxes
            gt_counts[class_id] += int(class_boxes.shape[0])
        ignore_by_image.append(
            _as_numpy_boxes(target.get("ignore_boxes", torch.empty((0, 4))))
        )

    for image_index, prediction in enumerate(predictions):
        boxes = _as_numpy_boxes(prediction["boxes"])
        labels = prediction["labels"].detach().to(device="cpu").numpy().reshape(-1)
        scores = (
            prediction["scores"]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
            .reshape(-1)
        )
        order = np.argsort(-scores, kind="stable")[:max_detections_per_image]
        for index in order:
            class_id = int(labels[index])
            if class_id in gt_counts:
                prediction_by_class[class_id].append(
                    (float(scores[index]), image_index, boxes[index])
                )

    per_class_by_threshold: dict[int, list[float]] = defaultdict(list)
    ap_by_threshold_class: dict[tuple[float, int], float] = {}
    for threshold in iou_thresholds:
        for class_id, ground_truth_count in gt_counts.items():
            if ground_truth_count == 0:
                continue
            detections = sorted(
                prediction_by_class[class_id], key=lambda item: item[0], reverse=True
            )
            matched_by_image: dict[int, set[int]] = defaultdict(set)
            true_positives: list[int] = []
            false_positives: list[int] = []
            for _, image_index, box in detections:
                ground_truth_boxes = gt_by_class[class_id][image_index]
                matched = matched_by_image[image_index]
                ious = _iou_with_box(box, ground_truth_boxes)
                if matched and ious.size:
                    ious[list(matched)] = -1.0
                best_index = int(np.argmax(ious)) if ious.size else -1
                if best_index >= 0 and ious[best_index] >= threshold:
                    matched.add(best_index)
                    true_positives.append(1)
                    false_positives.append(0)
                    continue
                ignored_ious = _iou_with_box(box, ignore_by_image[image_index])
                if ignored_ious.size and float(ignored_ious.max()) >= threshold:
                    continue
                true_positives.append(0)
                false_positives.append(1)
            ap = _average_precision(
                true_positives, false_positives, ground_truth_count, recall_points
            )
            per_class_by_threshold[class_id].append(ap)
            ap_by_threshold_class[(threshold, class_id)] = ap

    class_aps = {
        VISDRONE_CLASSES[class_id - 1]: float(np.mean(aps))
        for class_id, aps in per_class_by_threshold.items()
    }
    iou_50 = min(iou_thresholds, key=lambda value: abs(value - 0.5))
    if abs(iou_50 - 0.5) > 1e-8:
        raise ValueError("iou_thresholds must include 0.5 to report mAP@0.5")
    class_ap_50 = {
        VISDRONE_CLASSES[class_id - 1]: ap_by_threshold_class[(iou_50, class_id)]
        for class_id in per_class_by_threshold
    }
    return DetectionMetrics(
        map_50_95=float(np.mean(list(class_aps.values()))) if class_aps else 0.0,
        map_50=float(np.mean(list(class_ap_50.values()))) if class_ap_50 else 0.0,
        per_class_map_50_95=class_aps,
        per_class_map_50=class_ap_50,
    )
