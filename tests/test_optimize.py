from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from optimize.benchmark import (
    _compare_logits,
    _percentile,
    _prune_model,
    _quantize_model,
    _write_summary,
)
from optimize.config import BenchmarkConfig
from torch import nn


def test_benchmark_config_rejects_invalid_settings() -> None:
    cfg = BenchmarkConfig(
        seed=1,
        train_config="train.yaml",
        checkpoint="model.pt",
        output_dir="bench",
        warmup_iterations=0,
        benchmark_iterations=1,
        num_threads=1,
        pruning_amount=0.5,
        quantization_backend="qnnpack",
    )

    with pytest.raises(ValueError, match="benchmark_iterations"):
        cfg.validate()


def test_pruning_removes_masks_and_keeps_requested_sparsity() -> None:
    model = nn.Sequential(nn.Linear(4, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4))

    sparsity = _prune_model(model, amount=0.5)

    assert sparsity == 0.5
    assert "weight_orig" not in model[0]._parameters
    assert "weight_mask" not in model[0]._buffers


def test_dynamic_int8_quantization_runs_on_cpu() -> None:
    engines = torch.backends.quantized.supported_engines
    backend = next(
        engine for engine in ("qnnpack", "x86", "fbgemm") if engine in engines
    )
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 2)).eval()
    images = torch.rand(4, 3, 2, 2)

    quantized = _quantize_model(model, backend=backend)

    with torch.inference_mode():
        logits = quantized(images[:1])
    assert logits.shape == (1, 2)
    assert any(
        "quantized.dynamic" in type(module).__module__ for module in quantized.modules()
    )


def test_percentile_interpolates_between_samples() -> None:
    assert _percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.5


def test_logits_comparison_reports_top1_and_cosine() -> None:
    reference = np.array([[1.0, 0.0], [0.0, 2.0]])
    candidate = np.array([[1.0, 1.0], [3.0, 1.0]])

    top1_agreement, cosine_similarity = _compare_logits(reference, candidate)

    assert top1_agreement == 0.5
    assert cosine_similarity == pytest.approx((1 / np.sqrt(2) + 1 / np.sqrt(10)) / 2)


def test_benchmark_summary_writes_markdown_csv_and_raw_json(tmp_path) -> None:
    benchmark_cfg = BenchmarkConfig(
        seed=13,
        train_config="train.yaml",
        checkpoint="model.pt",
        output_dir=str(tmp_path),
        warmup_iterations=0,
        benchmark_iterations=2,
        num_threads=1,
        pruning_amount=0.5,
        quantization_backend="qnnpack",
    )
    row = {
        "variant": "baseline",
        "val_acc": 0.75,
        "top1_agreement": 1.0,
        "cosine_similarity": 1.0,
        "val_loss": 0.5,
        "val_samples": 4,
        "p50_ms": 1.0,
        "p99_ms": 2.0,
        "model_size_bytes": 1024,
        "peak_rss_bytes": 2048,
        "sparsity": 0.0,
        "artifact": "baseline.state_dict.pt",
    }

    _write_summary(tmp_path, [row], benchmark_cfg)

    assert (
        "| baseline | 0.7500 | 1.0000 | 1.000000 |"
        in (tmp_path / "results.md").read_text()
    )
    assert (
        "variant,val_acc,top1_agreement,cosine_similarity"
        in (tmp_path / "results.csv").read_text()
    )
    raw = json.loads((tmp_path / "raw.json").read_text())
    assert raw["metadata"]["thread_count"] == 1
    assert raw["results"][0]["variant"] == "baseline"
