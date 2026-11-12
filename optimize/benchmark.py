from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic
from torch.nn.utils import prune
from torch.utils.data import DataLoader

from optimize.config import BenchmarkConfig
from train.config import TrainConfig
from train.data import build_dataloaders
from train.gitinfo import git_hash
from train.model import load_classifier
from train.seed import seed_everything

VARIANTS = ("baseline", "pruned", "int8", "pruned_int8")
RESULT_FIELDS = (
    "variant",
    "val_acc",
    "top1_agreement",
    "cosine_similarity",
    "p50_ms",
    "p99_ms",
    "model_size_bytes",
    "peak_rss_bytes",
    "sparsity",
)


def _load_model(train_cfg: TrainConfig, checkpoint: str) -> nn.Module:
    return load_classifier(
        model_name=train_cfg.model_name,
        num_classes=train_cfg.num_classes,
        pretrained=False,
        checkpoint=checkpoint,
    ).cpu()


def _prune_model(model: nn.Module, amount: float) -> float:
    parameters = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, nn.Conv2d | nn.Linear)
    ]
    if parameters and amount:
        prune.global_unstructured(
            parameters,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        # Remove the masks so the saved state dict is a normal dense model.
        for module, name in parameters:
            prune.remove(module, name)

    total = sum(module.weight.numel() for module, _ in parameters)
    zeros = sum(
        int(torch.count_nonzero(module.weight == 0).item()) for module, _ in parameters
    )
    return zeros / total if total else 0.0


def _quantize_model(
    model: nn.Module,
    backend: str,
) -> nn.Module:
    if backend not in torch.backends.quantized.supported_engines:
        raise ValueError(
            f"quantization backend {backend!r} is unavailable; supported: "
            f"{torch.backends.quantized.supported_engines}"
        )
    torch.backends.quantized.engine = backend
    # MobileNet is convolution-heavy. Dynamic int8 keeps convolutions in FP32
    # and quantizes Linear weights, which preserves accuracy on this CPU path.
    return quantize_dynamic(model.eval(), {nn.Linear}, dtype=torch.qint8).eval()


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _evaluate_and_save_logits(
    model: nn.Module, loader: DataLoader, path: Path
) -> dict[str, float | int]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    n = 0
    logits_parts = []
    with torch.inference_mode():
        for images, targets in loader:
            images = images.cpu()
            targets = targets.cpu()
            logits = model(images).float()
            total_loss += criterion(logits, targets).item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            n += targets.size(0)
            logits_parts.append(logits.cpu())
    if n == 0:
        raise ValueError("validation split is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, torch.cat(logits_parts).numpy())
    return {"val_loss": total_loss / n, "val_acc": correct / n, "val_samples": n}


def _compare_logits(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[float, float]:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError(
            "reference and candidate logits must have matching [N, C] shape"
        )
    if reference.shape[0] == 0:
        raise ValueError("cannot compare empty logits")

    top1_agreement = float(
        np.mean(np.argmax(reference, axis=1) == np.argmax(candidate, axis=1))
    )
    reference = reference.astype(np.float64, copy=False)
    candidate = candidate.astype(np.float64, copy=False)
    dot = np.sum(reference * candidate, axis=1)
    reference_norm = np.linalg.norm(reference, axis=1)
    candidate_norm = np.linalg.norm(candidate, axis=1)
    denominator = reference_norm * candidate_norm
    cosine = np.divide(
        dot,
        denominator,
        out=np.zeros_like(dot),
        where=denominator > 0,
    )
    cosine[(reference_norm == 0) & (candidate_norm == 0)] = 1.0
    return top1_agreement, float(np.mean(cosine))


def _peak_rss_bytes() -> int:
    # macOS reports bytes; Linux reports KiB for ru_maxrss.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def _run_worker(cfg: BenchmarkConfig, variant: str) -> dict[str, Any]:
    cfg.validate()
    train_cfg = TrainConfig.from_yaml(cfg.train_config)
    if not Path(cfg.checkpoint).is_file():
        raise FileNotFoundError(f"checkpoint not found: {cfg.checkpoint}")

    seed_everything(cfg.seed)
    torch.set_num_threads(cfg.num_threads)
    torch.set_num_interop_threads(cfg.num_threads)
    _, val_loader, _ = build_dataloaders(train_cfg)
    model = _load_model(train_cfg, cfg.checkpoint)

    pruned = variant in {"pruned", "pruned_int8"}
    quantized = variant in {"int8", "pruned_int8"}
    sparsity = _prune_model(model, cfg.pruning_amount) if pruned else 0.0
    if quantized:
        model = _quantize_model(
            model,
            cfg.quantization_backend,
        )

    parity_dir = Path(cfg.output_dir) / ".parity_logits"
    logits_path = parity_dir / f"{variant}.npy"
    val_metrics = _evaluate_and_save_logits(model, val_loader, logits_path)
    try:
        sample_images, _ = next(iter(val_loader))
    except StopIteration as exc:
        raise ValueError("validation split is empty") from exc
    sample = sample_images[:1].cpu()

    with torch.inference_mode():
        for _ in range(cfg.warmup_iterations):
            model(sample)
        timings_ms: list[float] = []
        for _ in range(cfg.benchmark_iterations):
            start = time.perf_counter_ns()
            model(sample)
            timings_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)

    artifact_dir = Path(cfg.output_dir) / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{variant}.state_dict.pt"
    torch.save(model.state_dict(), artifact_path)
    result = {
        "variant": variant,
        "val_acc": val_metrics["val_acc"],
        "val_loss": val_metrics["val_loss"],
        "val_samples": val_metrics["val_samples"],
        "p50_ms": _percentile(timings_ms, 50),
        "p99_ms": _percentile(timings_ms, 99),
        "model_size_bytes": artifact_path.stat().st_size,
        "peak_rss_bytes": _peak_rss_bytes(),
        "sparsity": sparsity,
        "artifact": str(artifact_path),
        "parity_logits": str(logits_path),
    }
    return result


def _worker_main(config_path: str, variant: str) -> int:
    cfg = BenchmarkConfig.from_yaml(config_path)
    print(json.dumps(_run_worker(cfg, variant)))
    return 0


def _write_summary(
    output_dir: Path, rows: list[dict[str, Any]], cfg: BenchmarkConfig
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": cfg.seed,
        "git_hash": git_hash(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": platform.platform(),
        "processor": platform.processor() or os.uname().machine,
        "os": platform.system(),
        "python": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "torch": str(torch.__version__),
        "thread_count": cfg.num_threads,
        "warmup_iterations": cfg.warmup_iterations,
        "benchmark_iterations": cfg.benchmark_iterations,
        "method": (
            "CPU, batch=1, perf_counter_ns; validation accuracy, top-1 agreement, "
            "and cosine similarity"
        ),
        "checkpoint": cfg.checkpoint,
        "train_config": cfg.train_config,
        "quantization_backend": cfg.quantization_backend,
        "pruning_amount": cfg.pruning_amount,
        "quantization_method": "dynamic int8 for nn.Linear modules",
    }
    payload = {"metadata": metadata, "results": rows}
    (output_dir / "raw.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in RESULT_FIELDS} for row in rows)

    lines = [
        "# Classifier optimization benchmark",
        "",
        f"Hardware: {metadata['hardware']} ({metadata['processor']}); "
        f"threads: {cfg.num_threads}; warmup: {cfg.warmup_iterations}; "
        f"iterations: {cfg.benchmark_iterations}.",
        "",
        "Dynamic int8 quantizes Linear layers only; convolution layers remain FP32. "
        "Latency is per single CPU inference. Peak RSS is isolated per variant "
        "process. Pruned weights are saved as dense tensors; sparsity is shown "
        "separately because unstructured pruning alone does not shrink a dense file.",
        "",
        "| Variant | Top-1 accuracy | Top-1 agreement | Cosine similarity | "
        "p50 (ms) | p99 (ms) | State dict (bytes) | Peak RSS (bytes) | Sparsity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['val_acc']:.4f} | "
            f"{row['top1_agreement']:.4f} | {row['cosine_similarity']:.6f} | "
            f"{row['p50_ms']:.3f} | {row['p99_ms']:.3f} | "
            f"{row['model_size_bytes']} | {row['peak_rss_bytes']} | "
            f"{row['sparsity']:.4f} |"
        )
    (output_dir / "results.md").write_text("\n".join(lines) + "\n")


def run(config_path: str) -> list[dict[str, Any]]:
    cfg = BenchmarkConfig.from_yaml(config_path)
    cfg.validate()
    output_dir = Path(cfg.output_dir)
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        command = [
            sys.executable,
            "-m",
            "optimize.benchmark",
            "--worker",
            variant,
            "--config",
            config_path,
        ]
        worker_env = os.environ.copy()
        worker_env["OMP_NUM_THREADS"] = str(cfg.num_threads)
        worker_env["MKL_NUM_THREADS"] = str(cfg.num_threads)
        worker_env["OPENBLAS_NUM_THREADS"] = str(cfg.num_threads)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=worker_env,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"benchmark worker {variant!r} failed:\n{completed.stderr}\n"
                f"{completed.stdout}"
            )
        rows.append(json.loads(completed.stdout.strip().splitlines()[-1]))

    reference_logits = np.load(output_dir / ".parity_logits" / "baseline.npy")
    for row in rows:
        candidate_logits = np.load(row["parity_logits"])
        row["top1_agreement"], row["cosine_similarity"] = _compare_logits(
            reference_logits, candidate_logits
        )
        Path(row.pop("parity_logits")).unlink()
    (output_dir / ".parity_logits").rmdir()
    _write_summary(output_dir, rows, cfg)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline, pruned, int8, and pruned+int8 classifiers."
    )
    parser.add_argument("--config", default="optimize/configs/flowers102.yaml")
    parser.add_argument("--worker", choices=VARIANTS, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        return _worker_main(args.config, args.worker)
    rows = run(args.config)
    print(
        json.dumps(
            {
                "output_dir": BenchmarkConfig.from_yaml(args.config).output_dir,
                "variants": [row["variant"] for row in rows],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
