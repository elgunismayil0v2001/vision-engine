from __future__ import annotations

import numpy as np
import onnxruntime as ort
import torch
from torch import nn


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
