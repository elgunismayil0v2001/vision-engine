from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from train.detector_config import DetectorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the pinned VisDrone snapshot."
    )
    parser.add_argument("--config", default="train/configs/visdrone.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="download a seeded train subset; by default download the full train split",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=None,
        help=(
            "download a seeded validation subset; "
            "by default download the full val split"
        ),
    )
    return parser.parse_args()


def _select_split(
    samples: list[dict[str, Any]], split: str, limit: int | None, seed: int
) -> list[dict[str, Any]]:
    records = [sample for sample in samples if split in sample.get("tags", [])]
    if limit is None:
        return records
    if limit < 1:
        raise ValueError(f"{split} sample limit must be positive")
    if limit > len(records):
        raise ValueError(
            f"requested {limit} {split} samples, only {len(records)} exist"
        )
    return sorted(
        random.Random(seed).sample(records, limit), key=lambda row: row["filepath"]
    )


def download_dataset(
    cfg: DetectorConfig,
    *,
    train_samples: int | None = None,
    val_samples: int | None = None,
) -> Path:
    cfg.validate()
    data_root = Path(cfg.dataset_root)
    data_root.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id=cfg.dataset_repo,
            repo_type="dataset",
            revision=cfg.dataset_revision,
            local_dir=data_root,
            allow_patterns=["samples.json"],
            max_workers=cfg.download_workers,
        )
    )
    samples_path = snapshot / "samples.json"
    if not samples_path.is_file():
        raise FileNotFoundError(f"download completed without {samples_path}")
    payload: dict[str, Any] = json.loads(samples_path.read_text(encoding="utf-8"))
    records = payload.get("samples")
    if not isinstance(records, list):
        raise ValueError("VisDrone samples.json must contain a 'samples' list")
    train_records = _select_split(records, "train", train_samples, cfg.seed)
    val_records = _select_split(records, "val", val_samples, cfg.seed + 1)
    selected = train_records + val_records
    image_patterns = sorted({str(row["filepath"]).lstrip("/") for row in selected})
    snapshot_download(
        repo_id=cfg.dataset_repo,
        repo_type="dataset",
        revision=cfg.dataset_revision,
        local_dir=data_root,
        allow_patterns=image_patterns,
        max_workers=cfg.download_workers,
    )
    missing = [
        str(data_root / image)
        for image in image_patterns
        if not (data_root / image).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"download is missing {len(missing)} image files; first: {missing[0]}"
        )
    print(
        json.dumps(
            {
                "dataset_root": str(snapshot),
                "revision": cfg.dataset_revision,
                "samples_json": str(samples_path),
                "downloaded_train_samples": len(train_records),
                "downloaded_val_samples": len(val_records),
                "downloaded_images": len(image_patterns),
            }
        )
    )
    return snapshot


def main() -> int:
    args = parse_args()
    cfg = DetectorConfig.from_yaml(args.config)
    if args.data_root is not None:
        cfg = replace(cfg, dataset_root=args.data_root)
    download_dataset(
        cfg, train_samples=args.train_samples, val_samples=args.val_samples
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
