# Classifier optimization benchmark

Hardware: macOS-27.2.1-arm64-arm-64bit-Mach-O (arm); threads: 1; warmup: 20; iterations: 200.

Dynamic int8 quantizes Linear layers only; convolution layers remain FP32. Latency is per single CPU inference. Peak RSS is isolated per variant process. Pruned weights are saved as dense tensors; sparsity is shown separately because unstructured pruning alone does not shrink a dense file.

| Variant | Top-1 accuracy | Top-1 agreement | Cosine similarity | p50 (ms) | p99 (ms) | State dict (bytes) | Peak RSS (bytes) | Sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.7422 | 1.0000 | 1.000000 | 14.840 | 83.583 | 6622634 | 545423360 | 0.0000 |
| pruned | 0.0608 | 0.0676 | 0.805185 | 15.081 | 30.044 | 6622138 | 540753920 | 0.5000 |
| int8 | 0.7431 | 0.9951 | 0.999989 | 14.143 | 18.244 | 4540322 | 537493504 | 0.0000 |
| pruned_int8 | 0.0608 | 0.0667 | 0.805332 | 14.086 | 17.388 | 4542150 | 567017472 | 0.5000 |
