from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


def _run_preprocessing_parity(binary: Path, root: Path) -> None:
    pixels = np.array(
        [
            [[0, 127, 255], [255, 64, 32], [18, 90, 201]],
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
        ],
        dtype=np.uint8,
    )
    rgb_path = root / "rgb.bin"
    output_path = root / "preprocessed.f32"
    pixels.tofile(rgb_path)
    subprocess.run(
        [
            str(binary),
            "preprocess",
            str(pixels.shape[1]),
            str(pixels.shape[0]),
            str(rgb_path),
            str(output_path),
        ],
        check=True,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    expected = transform(Image.fromarray(pixels)).numpy().astype(np.float32).reshape(-1)
    actual = np.fromfile(output_path, dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def _run_model_parity(binary: Path, allocation_binary: Path, root: Path) -> None:
    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Flatten(), nn.Linear(3 * 4 * 4, 7), nn.Tanh(), nn.Linear(7, 5)
    ).eval()
    model_path = root / "toy_classifier.onnx"
    dummy = torch.zeros(1, 3, 4, 4)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            model_path,
            opset_version=17,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=False,
        )
    onnx.checker.check_model(model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    for batch in (1, 2):
        inputs = torch.randn(batch, 3, 4, 4)
        with torch.no_grad():
            expected = model(inputs).numpy()
        ort_logits = session.run(["logits"], {"input": inputs.numpy()})[0]
        np.testing.assert_allclose(ort_logits, expected, rtol=1e-4, atol=1e-5)

        input_path = root / f"input-{batch}.f32"
        output_path = root / f"cpp-logits-{batch}.f32"
        inputs.numpy().astype(np.float32).tofile(input_path)
        subprocess.run(
            [
                str(binary),
                "infer",
                str(model_path),
                "4",
                "5",
                str(batch),
                str(input_path),
                str(output_path),
            ],
            check=True,
        )
        cpp_logits = np.fromfile(output_path, dtype=np.float32).reshape(batch, 5)
        np.testing.assert_allclose(cpp_logits, expected, rtol=1e-4, atol=1e-5)

    subprocess.run(
        [str(allocation_binary), str(model_path), "4", "5", "2"],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--allocation-binary", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="vision-engine-parity-") as temporary:
        root = Path(temporary)
        _run_preprocessing_parity(args.binary.resolve(), root)
        _run_model_parity(args.binary.resolve(), args.allocation_binary.resolve(), root)
    print(
        "preprocessing and PyTorch/ONNX Runtime/C++ parity passed "
        "for batch sizes 1 and 2"
    )


if __name__ == "__main__":
    main()
