# vision-engine

Real-time aerial object detection engine, built in two layers:

1. **Core (weeks 1–3):** train a small classifier, export ONNX, run it from C++20, then quantise/prune and benchmark.
2. **Vision (weeks 4–7):** fine-tune a torchvision detector on VisDrone, add preprocessing kernels and a GPU video path.

Primary machine is a Mac M2 Air (no CUDA). GPU work is Colab/Kaggle later; every CUDA path will have a CPU fallback. VisDrone is treated as **non-commercial** (licence text is inconsistent). Do not use Ultralytics YOLO.

See [AGENTS.md](AGENTS.md) for milestones and engineering rules.

## Current milestone: week 1 (baseline classifier)

Train MobileNetV3-Small on Flowers102 with a fixed seed. Metrics and git hash are written next to the checkpoint.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m train --config train/configs/baseline.yaml
```

Outputs land in `outputs/flowers102_mobilenetv3_small/` (`metrics.jsonl`, `run.json`, `best.pt`).

Tests (no Flowers102 download; synthetic tensors):

```bash
pytest
ruff check train tests
black --check train tests
```

## Status

Week 1 is in progress: the training command and tests exist; a full Flowers102 run has not been claimed until it is executed on this machine.
