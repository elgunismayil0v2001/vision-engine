from __future__ import annotations

import argparse
import json
import os
import platform
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from train.detection_metrics import compute_map
from train.detector_config import DetectorConfig
from train.gitinfo import git_hash
from train.seed import seed_everything
from train.visdrone_data import (
    VisDroneDetectionDataset,
    detection_collate_fn,
    load_visdrone_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the VisDrone detector.")
    parser.add_argument("--config", default="train/configs/visdrone.yaml")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def build_detector(cfg: DetectorConfig, *, use_coco_weights: bool) -> torch.nn.Module:
    weights = (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
        if use_coco_weights
        else None
    )
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=(
            cfg.trainable_backbone_layers if weights is not None else None
        ),
        box_score_thresh=cfg.score_threshold,
        box_detections_per_img=cfg.max_detections_per_image,
    )
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, cfg.num_classes)
    return model


def _build_datasets(
    cfg: DetectorConfig,
) -> tuple[VisDroneDetectionDataset, VisDroneDetectionDataset]:
    samples = load_visdrone_samples(cfg.dataset_root)
    train_data = VisDroneDetectionDataset(
        cfg.dataset_root,
        "train",
        seed=cfg.seed,
        horizontal_flip_probability=cfg.horizontal_flip_probability,
        max_samples=cfg.max_train_samples,
        samples=samples,
    )
    val_data = VisDroneDetectionDataset(
        cfg.dataset_root,
        "val",
        seed=cfg.seed,
        max_samples=cfg.max_val_samples,
        samples=samples,
    )
    return train_data, val_data


def _resolve_device(preference: str) -> torch.device:
    if preference == "auto":
        # torchvision's MPS detection kernels are not reliable on this Mac; use
        # CUDA when present and the tested CPU fallback otherwise.
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError(f"requested detector device is unavailable: {preference}")


def _seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy in a worker; top-level so macOS spawn can pickle it."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_loader(
    dataset: VisDroneDetectionDataset,
    cfg: DetectorConfig,
    *,
    epoch_seed: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(epoch_seed)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _move_targets(
    targets: tuple[dict[str, torch.Tensor], ...], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    return [
        {
            key: value.to(device, non_blocking=True)
            for key, value in target.items()
            if key != "ignore_boxes"
        }
        for target in targets
    ]


@torch.inference_mode()
def evaluate_detector(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: DetectorConfig,
) -> dict[str, object]:
    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []
    for images, batch_targets in loader:
        images_on_device = [image.to(device, non_blocking=True) for image in images]
        batch_predictions = model(images_on_device)
        predictions.extend(batch_predictions)
        targets.extend(batch_targets)
    return compute_map(
        predictions,
        targets,
        iou_thresholds=cfg.iou_thresholds,
        recall_points=cfg.recall_points,
        max_detections_per_image=cfg.max_detections_per_image,
    ).to_dict()


def _rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: DetectorConfig,
    epoch: int,
    batches_completed: int,
    global_step: int,
    epoch_loss_sum: float,
    epoch_batches_count: int,
    best_map: float,
    metrics_history: list[dict[str, Any]],
    run_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": cfg.to_dict(),
        "epoch": epoch,
        "batches_completed": batches_completed,
        "global_step": global_step,
        "epoch_loss_sum": epoch_loss_sum,
        "epoch_batches_count": epoch_batches_count,
        "best_map_50_95": best_map,
        "metrics_history": metrics_history,
        "run_meta": run_meta,
        "rng_state": _rng_state(),
    }


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _check_resume_config(saved: dict[str, Any], cfg: DetectorConfig) -> None:
    mutable_fields = {
        "dataset_root",
        "epochs",
        "num_workers",
        "download_workers",
        "checkpoint_every_steps",
        "output_dir",
    }
    current = cfg.to_dict()
    for key, value in saved.items():
        if key not in mutable_fields and current.get(key) != value:
            raise ValueError(f"resume config mismatch for {key}")


def evaluate_checkpoint(
    cfg: DetectorConfig,
    checkpoint_path: Path,
    *,
    device_preference: str = "auto",
) -> dict[str, object]:
    cfg.validate()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    device = _resolve_device(device_preference)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_detector(cfg, use_coco_weights=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    _, val_data = _build_datasets(cfg)
    loader = _make_loader(
        val_data,
        cfg,
        epoch_seed=cfg.seed,
        shuffle=False,
        device=device,
    )
    return evaluate_detector(model, loader, device, cfg)


def train_detector(
    cfg: DetectorConfig,
    *,
    resume_path: Path | None = None,
    device_preference: str = "auto",
) -> dict[str, Any]:
    cfg.validate()
    seed_everything(cfg.seed)
    torch.set_num_threads(cfg.torch_threads)
    device = _resolve_device(device_preference)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_payload: dict[str, Any] | None = None
    if resume_path is not None:
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        _check_resume_config(resume_payload["config"], cfg)

    model = build_detector(
        cfg, use_coco_weights=cfg.use_coco_weights and resume_payload is None
    )
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma
    )

    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        start_epoch = int(resume_payload["epoch"])
        batches_to_skip = int(resume_payload["batches_completed"])
        global_step = int(resume_payload["global_step"])
        resume_epoch_loss = float(resume_payload.get("epoch_loss_sum", 0.0))
        resume_epoch_batches = int(resume_payload.get("epoch_batches_count", 0))
        best_map = float(resume_payload["best_map_50_95"])
        metrics_history = list(resume_payload["metrics_history"])
        run_meta = dict(resume_payload["run_meta"])
    else:
        start_epoch = 1
        batches_to_skip = 0
        global_step = 0
        resume_epoch_loss = 0.0
        resume_epoch_batches = 0
        best_map = -1.0
        metrics_history = []
        run_meta = {
            "seed": cfg.seed,
            "git_hash": git_hash(),
            "device": str(device),
            "hardware": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "torch_threads": torch.get_num_threads(),
            "dataset_repo": cfg.dataset_repo,
            "dataset_revision": cfg.dataset_revision,
            "dataset_license_policy": "non-commercial; share-alike",
            "config": cfg.to_dict(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    train_data, val_data = _build_datasets(cfg)
    run_meta["train_samples"] = len(train_data)
    run_meta["validation_samples"] = len(val_data)
    (output_dir / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n")
    metrics_path = output_dir / "metrics.jsonl"
    _write_metrics(metrics_path, metrics_history)
    final_metrics: dict[str, object] = metrics_history[-1] if metrics_history else {}
    resume_rng = resume_payload.get("rng_state") if resume_payload is not None else None

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_data.set_epoch(epoch)
        seed_everything(cfg.seed + epoch)
        train_loader = _make_loader(
            train_data,
            cfg,
            epoch_seed=cfg.seed + epoch,
            shuffle=True,
            device=device,
        )
        train_iterator = iter(train_loader)
        skip_count = batches_to_skip if epoch == start_epoch else 0
        if skip_count > len(train_loader):
            raise ValueError(
                "resume checkpoint batch position exceeds the epoch length"
            )
        for _ in range(skip_count):
            next(train_iterator)
        if epoch == start_epoch and resume_rng is not None:
            _restore_rng_state(resume_rng)
            resume_rng = None

        model.train()
        epoch_loss = resume_epoch_loss if epoch == start_epoch else 0.0
        epoch_batches = resume_epoch_batches if epoch == start_epoch else 0
        for batch_index in range(skip_count, len(train_loader)):
            images, batch_targets = next(train_iterator)
            images_on_device = [image.to(device, non_blocking=True) for image in images]
            targets_on_device = _move_targets(batch_targets, device)
            optimizer.zero_grad(set_to_none=True)
            loss_dict = model(images_on_device, targets_on_device)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite detector loss at step {global_step}"
                )
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            epoch_loss += loss_value
            epoch_batches += 1
            global_step += 1
            batches_completed = batch_index + 1

            if global_step % cfg.checkpoint_every_steps == 0:
                payload = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    epoch=epoch,
                    batches_completed=batches_completed,
                    global_step=global_step,
                    epoch_loss_sum=epoch_loss,
                    epoch_batches_count=epoch_batches,
                    best_map=best_map,
                    metrics_history=metrics_history,
                    run_meta=run_meta,
                )
                _atomic_save(payload, output_dir / "last.pt")
                print(
                    json.dumps(
                        {
                            "event": "checkpoint",
                            "epoch": epoch,
                            "step": global_step,
                            "loss": loss_value,
                            "path": str(output_dir / "last.pt"),
                        }
                    ),
                    flush=True,
                )

        val_loader = _make_loader(
            val_data,
            cfg,
            epoch_seed=cfg.seed + epoch,
            shuffle=False,
            device=device,
        )
        val_metrics = evaluate_detector(model, val_loader, device, cfg)
        mean_train_loss = epoch_loss / max(epoch_batches, 1)
        row: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": mean_train_loss,
            **val_metrics,
            "seed": cfg.seed,
            "git_hash": run_meta["git_hash"],
        }
        metrics_history.append(row)
        final_metrics = row
        scheduler.step()
        next_payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            epoch=epoch + 1,
            batches_completed=0,
            global_step=global_step,
            epoch_loss_sum=0.0,
            epoch_batches_count=0,
            best_map=max(best_map, float(val_metrics["map_50_95"])),
            metrics_history=metrics_history,
            run_meta=run_meta,
        )
        _atomic_save(next_payload, output_dir / "last.pt")
        if float(val_metrics["map_50_95"]) >= best_map:
            best_map = float(val_metrics["map_50_95"])
            next_payload["best_map_50_95"] = best_map
            _atomic_save(next_payload, output_dir / "best.pt")
        _write_metrics(metrics_path, metrics_history)
        print(json.dumps(row), flush=True)
        batches_to_skip = 0
        resume_epoch_loss = 0.0
        resume_epoch_batches = 0

    summary = {
        **run_meta,
        "best_map_50_95": best_map,
        "final": final_metrics,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    args = parse_args()
    cfg = DetectorConfig.from_yaml(args.config)
    if args.data_root is not None:
        cfg = replace(cfg, dataset_root=args.data_root)
    if args.output_dir is not None:
        cfg = replace(cfg, output_dir=args.output_dir)
    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--eval-only requires --checkpoint")
        result = evaluate_checkpoint(
            cfg, args.checkpoint, device_preference=args.device
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.checkpoint is not None:
        raise ValueError(
            "use --resume for training checkpoints; --checkpoint is eval-only"
        )
    summary = train_detector(
        cfg, resume_path=args.resume, device_preference=args.device
    )
    print(
        json.dumps(
            {
                "best_map_50_95": summary["best_map_50_95"],
                "output_dir": cfg.output_dir,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
