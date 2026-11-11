# AGENTS.md — vision-engine

Instructions for coding agents (Claude Code, Codex, Cursor, etc.) working in this repo.
The human owner is a quant-finance graduate learning ML deployment. Prefer clear, small,
well-tested steps over clever ones, and explain non-obvious choices in comments or the PR text.

## Project goal

A real-time aerial object detection engine in two layers, built to demonstrate C++20, PyTorch,
deployment, optimisation and benchmarking discipline.

- **Core layer (weeks 1-3), model-agnostic:** train a small classifier in PyTorch, export to ONNX,
  run it from a C++20 engine, optimise (int8 quantisation, pruning), benchmark honestly.
- **Vision layer (weeks 4-7):** fine-tune a detector on drone imagery (VisDrone2019-DET), add
  OpenMP / ARM NEON / CUDA image-preprocessing kernels, a GPU video pipeline with CUDA streams
  and pinned memory, and CPU-only CI.

## Hardware and environment constraints (important)

- Primary dev machine: **Mac M2 Air, no CUDA GPU**. Everything must build and test on macOS arm64
  and Linux x86_64 **without CUDA**.
- GPU work runs on **free Colab T4** (or Kaggle). Sessions are time-limited and ephemeral: keep
  data and checkpoints in Drive, make training resumable, never assume a session survives.
- **Every CUDA code path needs a CPU fallback with tests.** CUDA is behind a CMake option
  (`-DENABLE_CUDA=ON`, default OFF). CI is CPU-only.
- Docker on Mac has no CUDA; do not add GPU steps to Docker or CI.
- Do not install packages without a reason. Pin versions in `requirements.txt` / CMake.

## Repo layout

```
vision-engine/
  train/       config-driven training and evaluation (PyTorch)
  export/      ONNX export and parity tests
  optimize/    quantisation and pruning experiments
  cpp/         engine, preprocessing, benchmark harness, GoogleTest tests
  bench/       result tables, plots and raw logs
  docker/      Linux build image
  .github/workflows/   CPU-only CI
```

## Build milestones (work in this order)

| Week | Build | Done when |
|---|---|---|
| 1 | Train baseline classifier (MobileNet-class, Oxford-IIIT Pet or Flowers102 via `torchvision.datasets`) | One command reproduces training; fixed seeds; metrics logged |
| 2 | ONNX export + C++ run | PyTorch vs ONNX Runtime vs C++ outputs match within tolerance |
| 3 | Quantisation + pruning + benchmark harness | One results table: accuracy, p50/p99 latency, size, peak memory |
| 4 | Detector: `fasterrcnn_mobilenet_v3_large_320_fpn` (COCO_V1 weights) fine-tuned on VisDrone | mAP reported on the validation split |
| 5 | Image kernels: OpenMP, NEON, CUDA preprocessing | Each kernel matches the reference; speedups measured |
| 6 | GPU video path: CUDA streams, pinned memory, copy/compute overlap | FPS and p99 per stage reported |
| 7 | CI, Docker, README, write-up | CI green on Ubuntu; README has results + limitations |

Do not start a later milestone until the current one's "done when" holds and its tests pass.

## Engineering rules

**Python**
- Python 3.10+, PyTorch, type hints, `ruff` + `black`. Config-driven (YAML/dataclass), no magic numbers in code.
- Seed everything (`random`, `numpy`, `torch`); log the seed and git hash with every run.
- ONNX export: dynamic batch axis, opset ~17, `eval()` mode, `torch.no_grad()`. Detector export is
  fiddly; if it fails, export the backbone/heads separately and document it.

**C++**
- C++20, CMake, GoogleTest, ONNX Runtime (CPU and optional CUDA execution providers).
- RAII everywhere; no raw `new`/`delete`; follow the rule of 0/5. Prefer `std::span`, `std::unique_ptr`.
- Build with `-Wall -Wextra -Wpedantic`; run tests under ASan + UBSan (and TSan for threaded code).
- **Memory discipline:** preallocate buffers and arenas at startup; **zero allocations in steady-state
  inference**, verified by an allocation counter in a test.
- **Low latency:** warm up before timing; report p50/p99/p99.9; fix thread counts; measure per stage.
- OpenMP loops: state the schedule and thread count; compare against the scalar reference.

**Parity tests (required, not optional)**
- PyTorch vs ONNX Runtime (Python), Python vs C++, preprocessing parity (same pixels in, same tensor out),
  metric parity. Quantised models: compare top-1 agreement and cosine similarity, not exact equality.

**Benchmarks**
- Always record hardware, OS, compiler, thread count, warmup, iterations, and method in the output.
- Save raw logs in `bench/`. Never quote a number that is not reproducible from a script in the repo.
- Do not claim to beat production kernels. Report what was measured and why.

## Data and licences

- Core: `torchvision.datasets` (Oxford-IIIT Pet or Flowers102).
- Vision: **VisDrone2019-DET** (official repo, or the Voxel51 Hugging Face copy, 8,629 samples).
  Licence information is inconsistent (CC BY-SA 3.0 vs CC BY-NC-SA 3.0): **treat as non-commercial**
  and say so in the README. Cite the VisDrone TPAMI 2021 paper.
- Fallback if terms don't suit: a COCO subset.
- **Do not use Ultralytics YOLO** (AGPL-3.0, enterprise licence needed for closed use). Use torchvision models.
- Never commit datasets or large checkpoints; keep them in Drive and `.gitignore` them.

## Workflow for agents

1. Read this file and the current milestone before changing anything.
2. Make the smallest change that moves the milestone forward; add or update tests in the same change.
3. Run: Python tests (`pytest`), C++ build + `ctest`, sanitizers, formatters. Report what actually ran.
4. If something fails or cannot be verified here (e.g. CUDA code on a machine without a GPU),
   say so plainly. Never claim a test passed, or a speedup exists, without running it.
5. Commit in small, descriptive steps. Do not rewrite history or touch unrelated files.
6. Ask the owner before: adding a heavy dependency, changing the repo layout, or changing a milestone's scope.

## Definition of done

- [ ] Training reproducible from one command, fixed seeds, logged metrics
- [ ] Parity tests pass: PyTorch vs ONNX Runtime, Python vs C++, preprocessing
- [ ] One results table covers all variants: accuracy, p50/p99 latency, size, peak memory
- [ ] Benchmarks state hardware, thread count, warmup and method; fixed settings
- [ ] Steady-state inference makes zero allocations, shown by a counter
- [ ] CI runs on Ubuntu with Docker, tests and sanitizers
- [ ] README shows the results table, architecture and an honest limitations section
- [ ] Vision: detector mAP reported on the validation split
- [ ] Vision: GPU pipeline reports FPS and p99 latency per stage
- [ ] Vision: every CUDA path has a CPU fallback with tests
- [ ] Optional stretch: custom operator passes `gradcheck` and is benchmarked against pure PyTorch
  (the OpenMP/CUDA preprocessing kernels are the main C++ performance work)
