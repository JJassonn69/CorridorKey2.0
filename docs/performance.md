# Inference performance

602 frames across 11 configurations passed zero-tolerance alpha/foreground
comparison; all 1,806 raw-array and EXR files matched their references by SHA-256.
The packaged runtime matches the qualified source byte for byte.

## Measurements

RTX A5000, Ryzen 5 2600, PyTorch 2.9.1+cu128, Triton 3.5.1. Each row compares
one accepted step with its own control; rows are separate experiments.

| Change | Metric | 720p, before → after | 1920×1088, before → after |
|---|---|---:|---:|
| Foundation reuse + offloading | Peak allocated GiB | 7.52 → 5.23 | 11.33 → 8.86 |
| Compact parameter storage + unused-head removal | Peak allocated GiB | 5.22 → 3.48 | 8.86 → 7.20 |
| Four EXR workers + input prefetch | Seconds | 59.71 → 24.47 | 142.23 → 60.42 |
| Byte-value lookup + six EXR workers | Seconds | 24.706 → 21.289 | 58.122 → 50.073 |
| Ordered-FMA depthwise kernel | Seconds | 21.387 → 20.906 | Unqualified |

Timing: 72 frames, eight-frame warmup, native decode through float EXR alpha/FG
export; loading and verification excluded. Speed rows use medians of two fresh
processes per variant in reversed order. Earlier rounds include thermal
slowdown; the final 720p pair has none. The final 1080p timing was excluded for
CPU contention and thermal slowdown. Triton compilation cache was populated.
Allocator peaks exclude loading/warmup and are not minimum GPU capacities.
These results do not establish a combined original-to-final speedup.

## Runtime behaviour

- Reuse lossless per-frame foundation features and move completed foundations
  to CPU. This adds a decode pass and temporary disk/host storage. Existing
  `--low_vram` mode remains available.
- Store eligible operator parameters in BF16; preserve CSwin, normalization,
  statistics and standalone scales at their original precision.
- Prefetch one input window. EXR queues bound running plus queued work to
  `--exr_workers + 2` batches per output (default six workers).
- Use the optional depthwise fast path only on PyTorch 2.9.1 / Triton 3.5.1 for
  native CUDA BF16 valid 7×7 convolution. Other backends, shapes, precision and
  gradient-enabled calls retain PyTorch convolution. First use compiles a kernel.

No retraining or checkpoint changes. Exactness covers the tested footage and
environment, including short/tail windows, hair, motion blur, transparency,
low-VRAM mode, linear EXRs, Stage 2 batching and sequence hints.

## Regression tests

From the repository root, in the configured runtime environment:

```sh
python -m unittest discover -s tests -v
```

17 tests passed on the test GPU, covering exact kernel arithmetic and fallbacks,
all 256 input byte values, foundation tails, bounded prefetch/export queues,
serial/parallel EXR byte equality and failure cleanup. CUDA tests skip when
unavailable; kernel qualification tests require the versions above.
