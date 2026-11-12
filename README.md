# vision-engine

Real-time aerial object detection engine, built in two layers:

1. **Core (weeks 1–3):** train a small classifier, export ONNX, run it from C++20, then quantise/prune and benchmark.
2. **Vision (weeks 4–7):** fine-tune a torchvision detector on VisDrone, add preprocessing kernels and a GPU video path.

Primary machine is a Mac M2 Air (no CUDA). GPU work is Colab/Kaggle later; every CUDA path will have a CPU fallback. VisDrone is treated as **non-commercial** (licence text is inconsistent). Do not use Ultralytics YOLO.

See [AGENTS.md](AGENTS.md) for milestones and engineering rules.

## Current milestone: week 5 (image preprocessing kernels)

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

The full Flowers102 baseline run and Week 3 CPU comparison are complete.

## Week 2: ONNX export and C++ inference

Export the Week 1 checkpoint with a dynamic batch axis, then run the C++20 CPU engine. The ONNX file and JSON sidecar are written under `outputs/` and are ignored by git.

```bash
pip install -r requirements-dev.txt
python -m export --config export/configs/export.yaml
```

Install the pinned Python packages through `requirements-dev.txt`. The C++ build also needs the ONNX Runtime 1.31.1 CPU C++ SDK for the host platform. For Apple Silicon:

```bash
curl -L --fail https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-osx-arm64-1.31.1.tgz -o /tmp/onnxruntime-osx-arm64-1.31.1.tgz
mkdir -p /tmp/onnxruntime-sdk
tar -xzf /tmp/onnxruntime-osx-arm64-1.31.1.tgz --strip-components=1 -C /tmp/onnxruntime-sdk
cmake -S cpp -B build/cpp \
  -DONNXRUNTIME_ROOT=/tmp/onnxruntime-sdk \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build/cpp
ctest --test-dir build/cpp --output-on-failure
```

For AddressSanitizer and UndefinedBehaviorSanitizer, repeat the CMake build with `-DENABLE_SANITIZERS=ON`.

For Linux x86_64, use the `onnxruntime-linux-x64-1.31.1.tgz` asset in place of the macOS SDK. CMake uses installed GoogleTest 1.15.2 or fetches that exact tag. The CTest suite checks RGB preprocessing against torchvision, then checks PyTorch, Python ONNX Runtime, and the C++ executable on dynamic batch sizes 1 and 2. The parity test uses a deterministic small model and does not download Flowers102. The runtime CLI accepts caller-provided float32 NCHW buffers and writes logits to a float32 file; its preprocessing command converts RGB uint8 pixels to the same ImageNet-normalized NCHW representation as Week 1.

The C++ engine allocates reusable input and output tensors at construction for batch sizes from 1 through the configured maximum, then copies to and from those buffers on each call. CTest warms batches 1 and 2 and counts global C++ `operator new` calls across 100 inference calls. On this Mac with ONNX Runtime 1.31.1, the synthetic model currently reports 24 calls per inference across the engine and runtime. That does not satisfy the project's zero-allocation target; the counter also cannot observe allocations made through C `malloc`. Use `ctest --verbose` to see the allocation probe output.

Recheck the exported Flowers102 checkpoint against PyTorch, Python ONNX Runtime, and C++:

```bash
python -m export.parity \
  --config export/configs/export.yaml \
  --cpp-binary build/cpp/vision_engine_cli
```

On the Mac M2 Air CPU, the trained checkpoint produced these maximum absolute logit differences for seeded random float32 inputs (tolerance: `rtol=1e-4`, `atol=1e-5`):

| Batch | PyTorch vs ONNX Runtime | PyTorch vs C++ |
|---:|---:|---:|
| 1 | 3.04e-6 | 2.58e-6 |
| 2 | 2.44e-6 | 2.44e-6 |

The CTest suite also compares the C++ RGB-to-NCHW normalizer with torchvision on identical uint8 pixels. These checks validate inference and normalization parity; they do not measure C++ accuracy on the full Flowers102 validation split.

## Week 4: VisDrone detector fine-tuning

The detector is torchvision's `fasterrcnn_mobilenet_v3_large_320_fpn` initialized
with `FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1`. Its final box
predictor is replaced with 10 VisDrone classes plus background. The dataset
loader converts the pinned Voxel51 export's normalized boxes into pixel `xyxy`
boxes, and keeps `ignore_regions` and `others` out of the loss and evaluation.
Validation reports class-averaged 101-point AP at IoU 0.50:0.05:0.95 and AP at
IoU 0.50.

Download the full train and validation splits, then train using the pinned
settings:

```bash
python -m train.download_visdrone --config train/configs/visdrone.yaml
python -m train.detector --config train/configs/visdrone.yaml --device auto
```

The downloader also supports seeded subsets. The local smoke run used 16
training images and all 548 validation images:

```bash
python -m train.download_visdrone --train-samples 16
python -m train.detector --config train/configs/visdrone_smoke.yaml --device cpu
```

That smoke run is a pipeline check, not a final detector-quality result. It
trained one epoch on 16 of 6,471 train images; its low mAP reflects this small
training sample. The committed config uses all 6,471 train and 548 validation
images for 10 epochs. Checkpoints are written periodically and can be resumed
with `--resume`. For an ephemeral Colab session, mount Drive and pass the same
Drive paths as `--data-root` and `--output-dir` on both commands, then resume
from `<output-dir>/last.pt` after reconnecting. The automatic device selection
uses CUDA when available and CPU otherwise; `--device cpu` forces the fallback.

| Run | Train images | Epochs | Validation images | mAP@[.50:.95] | mAP@.50 |
|---|---:|---:|---:|---:|---:|
| Local smoke, seed 42 | 16 | 1 | 548 | 0.20% | 0.67% |

The raw run record, including per-class AP and hardware/software metadata, is [bench/visdrone/raw.json](bench/visdrone/raw.json). Local run metadata and checkpoints are also written under `outputs/visdrone2019_detector_cpu_smoke/`; datasets and model checkpoints are ignored by git. VisDrone is treated as non-commercial and share-alike because available license statements differ. The dataset mirror is [Voxel51/VisDrone2019-DET](https://huggingface.co/datasets/Voxel51/VisDrone2019-DET); cite Zhu et al., [“Detection and Tracking Meet Drones Challenge,” IEEE TPAMI (2021)](https://doi.org/10.1109/TPAMI.2021.3119563).

## Week 5: OpenMP, NEON, and CUDA preprocessing

The scalar RGB-to-NCHW normalizer is the reference. The OpenMP variant uses a
static schedule and a caller-specified thread count. ARM64 builds use NEON; other
architectures fall back to scalar. `ENABLE_CUDA` defaults to `OFF`. When enabled,
`CudaPreprocessor` allocates device buffers once and reuses them; when no CUDA
device is present or those allocations fail, it calls the scalar CPU path.

Build and run the CPU tests and benchmark on macOS or Linux:

```bash
cmake -S cpp -B build/cpp-week5 \
  -DONNXRUNTIME_ROOT=/tmp/onnxruntime-sdk \
  -DENABLE_CUDA=OFF -DENABLE_OPENMP=ON
cmake --build build/cpp-week5
ctest --test-dir build/cpp-week5 --output-on-failure
python bench/preprocess/run_benchmark.py \
  --binary build/cpp-week5/preprocess_benchmark \
  --output bench/preprocess/macos_m2.json \
  --width 640 --height 480 --warmup 20 --iterations 200 --threads 4
```

The Mac M2 Air run used Apple LLVM 21, macOS 27.2.1, a 640×480 deterministic
input, 20 warmups, and 200 timed calls. NEON output matched the scalar reference
within 1e-6. OpenMP was not available in this local toolchain, so its row is a
scalar fallback and is not an OpenMP speed measurement.

| Kernel | Effective threads | p50 (ms) | p99 (ms) | Speedup vs scalar |
|---|---:|---:|---:|---:|
| Scalar reference | 1 | 4.635 | 5.560 | 1.00× |
| OpenMP unavailable; scalar fallback | 1 | 4.703 | 6.592 | 0.99× |
| ARM NEON | 1 | 4.518 | 5.722 | 1.03× |

Hardware, compiler, git revision, warmup/iteration settings, p99.9, and parity
error are in the [raw Mac run record](bench/preprocess/macos_m2.json).

For the CUDA build and T4 measurements, open [week5_colab.ipynb](bench/preprocess/week5_colab.ipynb)
in Colab, select a T4 runtime, set the Drive repository path, then run all cells.
It builds with `-DENABLE_CUDA=ON`, checks parity and CPU fallback, and writes
`bench/preprocess/colab_t4.json` to Drive. CUDA timing includes both host/device
copies and uses preallocated buffers. This workspace has no connected Colab
runtime, so no CUDA speedup is reported here; the Week 5 GPU measurement remains
to be run in that notebook.

Tests (synthetic tensors; no Flowers102 download):

```bash
pytest
ruff check train export optimize tests
black --check train export optimize tests
```

## Status

Week 1's full Flowers102 run, Week 2's ONNX/C++ inference and parity gate, and Week 3's measured optimization table are complete. Week 4's resumable VisDrone pipeline and validation metric are implemented; its local smoke run used 16 training images, and the configured full training remains for Colab/Kaggle. Week 5's CPU paths, parity tests, and Mac NEON measurement are complete. Its OpenMP measurement requires an OpenMP toolchain, and its CUDA parity test and T4 measurement remain to be run in Colab.
