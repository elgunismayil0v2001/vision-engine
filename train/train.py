from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from train.config import TrainConfig
from train.data import build_dataloaders
from train.device import pick_device
from train.evaluate import evaluate
from train.gitinfo import git_hash
from train.model import build_model, count_parameters, save_checkpoint
from train.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the week-1 classifier.")
    parser.add_argument(
        "--config",
        default="train/configs/baseline.yaml",
        help="YAML config path",
    )
    return parser.parse_args()


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def run(cfg: TrainConfig) -> dict:
    seed_everything(cfg.seed)
    device = pick_device()
    run_hash = git_hash()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, n_train = build_dataloaders(cfg)
    model = build_model(cfg.model_name, cfg.num_classes, cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    run_meta = {
        "seed": cfg.seed,
        "git_hash": run_hash,
        "device": str(device),
        "hardware": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "params": count_parameters(model),
        "n_train": n_train,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg.to_dict(),
    }
    (output_dir / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n")

    metrics_path = output_dir / "metrics.jsonl"
    best_acc = -1.0
    last_metrics: dict[str, float] = {}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += targets.size(0)

        train_loss = running_loss / max(seen, 1)
        train_acc = correct / max(seen, 1)
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["val_loss"],
            "val_acc": val_metrics["val_acc"],
            "seed": cfg.seed,
            "git_hash": run_hash,
        }
        _append_jsonl(metrics_path, row)
        last_metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            **val_metrics,
        }

        ckpt_path = output_dir / "last.pt"
        save_checkpoint(
            str(ckpt_path),
            model,
            epoch,
            last_metrics,
            {"seed": cfg.seed, "git_hash": run_hash, "config": cfg.to_dict()},
        )
        if val_metrics["val_acc"] >= best_acc:
            best_acc = val_metrics["val_acc"]
            save_checkpoint(
                str(output_dir / "best.pt"),
                model,
                epoch,
                last_metrics,
                {"seed": cfg.seed, "git_hash": run_hash, "config": cfg.to_dict()},
            )

    summary = {
        **run_meta,
        "best_val_acc": best_acc,
        "final": last_metrics,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    args = parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    summary = run(cfg)
    print(
        json.dumps(
            {
                "best_val_acc": summary["best_val_acc"],
                "output_dir": cfg.output_dir,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
