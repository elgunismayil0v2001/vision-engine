from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import onnx
import torch

from export.config import ExportConfig
from train.gitinfo import git_hash
from train.model import load_classifier
from train.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the classifier to ONNX.")
    parser.add_argument(
        "--config",
        default="export/configs/export.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override checkpoint path",
    )
    parser.add_argument(
        "--onnx-path",
        default=None,
        help="Override ONNX output path",
    )
    return parser.parse_args()


def export_onnx(cfg: ExportConfig) -> Path:
    seed_everything(cfg.seed)
    ckpt = cfg.checkpoint
    if ckpt is not None and not Path(ckpt).is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    model = load_classifier(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained if ckpt is None else False,
        checkpoint=ckpt,
    )
    model.eval()

    dummy = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    out_path = Path(cfg.onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Dynamic batch so C++ / Python can run N=1 in tests and N>1 later.
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            opset_version=cfg.opset,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=False,
        )

    onnx.checker.check_model(out_path)

    sidecar = {
        "onnx_path": str(out_path),
        "checkpoint": ckpt,
        "seed": cfg.seed,
        "git_hash": git_hash(),
        "opset": cfg.opset,
        "input_name": "input",
        "output_name": "logits",
        "layout": "NCHW",
        "preprocessing": {
            "color_space": "RGB",
            "scale": 1.0 / 255.0,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "image_size": cfg.image_size,
        "num_classes": cfg.num_classes,
    }
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return out_path


def main() -> int:
    args = parse_args()
    cfg = ExportConfig.from_yaml(args.config)
    if args.checkpoint is not None:
        cfg = replace(cfg, checkpoint=args.checkpoint)
    if args.onnx_path is not None:
        cfg = replace(cfg, onnx_path=args.onnx_path)
    path = export_onnx(cfg)
    print(json.dumps({"onnx_path": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
