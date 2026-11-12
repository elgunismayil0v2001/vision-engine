# vision-engine

Real-time aerial object detection engine, built in two layers:

1. **Core (weeks 1–3):** train a small classifier, export ONNX, run it from C++20, then quantise/prune and benchmark.
2. **Vision (weeks 4–7):** fine-tune a torchvision detector on VisDrone, add preprocessing kernels and a GPU video path.

Primary machine is a Mac M2 Air (no CUDA). GPU work is Colab/Kaggle later; every CUDA path will have a CPU fallback. VisDrone is treated as **non-commercial** (licence text is inconsistent). Do not use Ultralytics YOLO.

See [AGENTS.md](AGENTS.md) for milestones and engineering rules.

## Current milestone: week 3 (quantisation, pruning, and benchmark harness)

The Week 3 harness compares the trained baseline with globally pruned, dynamic int8, and pruned+int8 variants. It evaluates top-1 accuracy on validation images and benchmarks CPU batch-1 latency in isolated processes.

```bash
python -m optimize --config optimize/configs/flowers102.yaml
```

The command expects `outputs/flowers102_mobilenetv3_small/best.pt` from the Week 1 training run. It writes `results.md`, `results.csv`, and `raw.json` plus one state-dict artifact per variant under `bench/flowers102/`. `raw.json` records the git hash, hardware, software versions, fixed thread count, warmup, and iteration count. The reported memory is peak process RSS, measured in a fresh process for each variant.

The report includes top-1 agreement and cosine similarity against baseline logits for each variant, including int8. Dynamic int8 quantises the classifier's `Linear` layers; MobileNet's convolution layers remain float32. Pruning removes the temporary masks before saving, so pruned weights stay dense and the file may not shrink; the table reports zero-weight sparsity separately.

### Flowers102 results

Measured on the Mac M2 Air. The baseline was trained for five epochs on MPS; all four benchmark variants ran on CPU with one thread, 20 warmups, 200 timed batch-1 iterations, and 1,020 validation images. Exact metadata and values are in [raw.json](bench/flowers102/raw.json) and [results.csv](bench/flowers102/results.csv).

| Variant | Top-1 accuracy | Top-1 agreement | Cosine similarity | p50 (ms) | p99 (ms) | State dict (bytes) | Peak RSS (bytes) | Sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 74.22% | 100.00% | 1.000000 | 14.840 | 83.583 | 6,622,634 | 545,423,360 | 0% |
| 50% pruned | 6.08% | 6.76% | 0.805185 | 15.081 | 30.044 | 6,622,138 | 540,753,920 | 50% |
| Dynamic int8 | 74.31% | 99.51% | 0.999989 | 14.143 | 18.244 | 4,540,322 | 537,493,504 | 0% |
| 50% pruned + dynamic int8 | 6.08% | 6.67% | 0.805332 | 14.086 | 17.388 | 4,542,150 | 567,017,472 | 50% |

In this run, dynamic int8 reduced the saved state dict while retaining baseline accuracy and logits. Unstructured 50% pruning without fine-tuning severely damaged accuracy and did not reduce the dense state-dict size. The measurements are specific to this run and hardware; they are not a claim about production performance.

Set up the pinned environment and reproduce the baseline checkpoint first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m train --config train/configs/baseline.yaml
```

The full Flowers102 baseline run and Week 3 CPU comparison are complete. The current checkout has not completed Week 2's C++ inference and parity gate.

Tests (synthetic tensors; no Flowers102 download):

```bash
pytest
ruff check train optimize tests
black --check train optimize tests
```

## Status

Week 1's full Flowers102 run and Week 3's measured optimization table are complete. Week 2's C++ inference/parity work remains incomplete in this checkout.
