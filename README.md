# vision-engine

Real-time aerial object detection engine, built in two layers:

1. **Core (weeks 1–3):** train a small classifier, export ONNX, run it from C++20, then quantise/prune and benchmark.
2. **Vision (weeks 4–7):** fine-tune a torchvision detector on VisDrone, add preprocessing kernels and a GPU video path.

Primary machine is a Mac M2 Air (no CUDA). GPU work is Colab/Kaggle later; every CUDA path will have a CPU fallback. VisDrone is treated as **non-commercial** (licence text is inconsistent). Do not use Ultralytics YOLO.

See [AGENTS.md](AGENTS.md) for milestones and engineering rules.

## Current milestone: week 3 (quantisation, pruning, and benchmark harness)

The Week 3 harness compares the trained baseline with globally pruned, static int8, and pruned+int8 variants. It calibrates int8 on training images, evaluates top-1 accuracy on validation images, and benchmarks CPU batch-1 latency in isolated processes.

```bash
python -m optimize --config optimize/configs/flowers102.yaml
```

The command expects `outputs/flowers102_mobilenetv3_small/best.pt` from the Week 1 training run. It writes `results.md`, `results.csv`, and `raw.json` plus one state-dict artifact per variant under `bench/flowers102/`. `raw.json` records the git hash, hardware, software versions, fixed thread count, warmup, and iteration count. The reported memory is peak process RSS, measured in a fresh process for each variant.

The report includes top-1 agreement and cosine similarity against baseline logits for each variant, including int8. Pruning removes the temporary masks before saving, so pruned weights stay dense and the file may not shrink; the table reports zero-weight sparsity separately. Static post-training int8 quantisation uses FX graph mode and the configured CPU backend.

Set up the pinned environment and reproduce the baseline checkpoint first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m train --config train/configs/baseline.yaml
```

Week 3 code and synthetic-data tests do not imply a Flowers102 result. The real benchmark remains pending until that baseline training run completes. The current checkout also has not completed Week 2's C++ inference and parity gate.

Tests (synthetic tensors; no Flowers102 download):

```bash
pytest
ruff check train optimize tests
black --check train optimize tests
```

## Status

Week 3's optimizer and benchmark harness are implemented. A measured Flowers102 comparison has not been claimed because the baseline checkpoint is absent. Week 1's full dataset run and Week 2's C++ inference/parity work remain unverified/incomplete in this checkout.
