from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from export.config import ExportConfig
from train.model import load_classifier

_RTOL = 1e-4
_ATOL = 1e-5


def torch_logits(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(x).cpu().numpy()


def ort_logits(onnx_path: str, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: x})[0]


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    va = a.reshape(-1).astype(np.float64)
    vb = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0.0:
        return 1.0 if np.allclose(va, vb) else 0.0
    return float(np.dot(va, vb) / denom)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the trained PyTorch, ONNX Runtime, and C++ classifiers."
    )
    parser.add_argument("--config", default="export/configs/export.yaml")
    parser.add_argument("--cpp-binary", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    return parser.parse_args()


def validate_cpp_parity(
    cfg: ExportConfig,
    cpp_binary: Path,
    batch_sizes: list[int],
) -> list[dict[str, float | int]]:
    if cfg.checkpoint is None:
        raise ValueError(
            "parity validation requires a trained checkpoint in the config"
        )
    if not Path(cfg.checkpoint).is_file():
        raise FileNotFoundError(f"checkpoint not found: {cfg.checkpoint}")
    onnx_path = Path(cfg.onnx_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}; export it first")
    if not cpp_binary.is_file():
        raise FileNotFoundError(f"C++ executable not found: {cpp_binary}")
    if not batch_sizes or any(batch <= 0 for batch in batch_sizes):
        raise ValueError("batch sizes must be positive")

    model = load_classifier(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        pretrained=False,
        checkpoint=cfg.checkpoint,
    )
    model.eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    results: list[dict[str, float | int]] = []

    with tempfile.TemporaryDirectory(prefix="flowers102-cpp-parity-") as temporary:
        temp_path = Path(temporary)
        for batch in batch_sizes:
            generator = torch.Generator(device="cpu").manual_seed(cfg.seed + batch)
            inputs = torch.randn(
                batch,
                3,
                cfg.image_size,
                cfg.image_size,
                generator=generator,
                dtype=torch.float32,
            )
            pytorch = torch_logits(model, inputs)
            onnx = session.run(["logits"], {"input": inputs.numpy()})[0]

            input_path = temp_path / f"input-{batch}.f32"
            cpp_output_path = temp_path / f"cpp-output-{batch}.f32"
            inputs.numpy().tofile(input_path)
            subprocess.run(
                [
                    str(cpp_binary.resolve()),
                    "infer",
                    str(onnx_path),
                    str(cfg.image_size),
                    str(cfg.num_classes),
                    str(batch),
                    str(input_path),
                    str(cpp_output_path),
                ],
                check=True,
            )
            cpp = np.fromfile(cpp_output_path, dtype=np.float32).reshape(
                batch, cfg.num_classes
            )

            torch_onnx_diff = max_abs_diff(pytorch, onnx)
            torch_cpp_diff = max_abs_diff(pytorch, cpp)
            if not np.allclose(pytorch, onnx, rtol=_RTOL, atol=_ATOL):
                raise AssertionError(f"PyTorch/ONNX mismatch at batch size {batch}")
            if not np.allclose(pytorch, cpp, rtol=_RTOL, atol=_ATOL):
                raise AssertionError(f"PyTorch/C++ mismatch at batch size {batch}")
            results.append(
                {
                    "batch": batch,
                    "pytorch_onnx_max_abs": torch_onnx_diff,
                    "pytorch_cpp_max_abs": torch_cpp_diff,
                    "pytorch_onnx_cosine": cosine_similarity(pytorch, onnx),
                    "pytorch_cpp_cosine": cosine_similarity(pytorch, cpp),
                }
            )
    return results


def main() -> int:
    args = parse_args()
    cfg = ExportConfig.from_yaml(args.config)
    results = validate_cpp_parity(cfg, args.cpp_binary, args.batch_sizes)
    print(json.dumps({"onnx_path": cfg.onnx_path, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
