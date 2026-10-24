# Private API pilot results — 2026-10-24

**Keep one worker as the default.** Two workers preserved exact outputs but
provided only 1.7% more end-to-end throughput and increased median job latency
10.2% under the observed load. Two remains a configurable, correctness-tested option.

Mac client → authenticated SSH/ProxyJump → private GPU API. RTX A5000,
23,028 MiB device memory, one host. Four fresh full jobs per configuration,
two independent users, 72 frames/job at 1024×578. Each client uploaded PNGs,
polled processing, downloaded all float EXRs and verified local bytes against
direct inference. Raw qualification capture was disabled for timing.

| Measured | One worker | Two workers |
|---|---:|---:|
| Four-job wall time | 289.0 s | 284.1 s |
| Combined end-to-end throughput | 0.997 fps | 1.014 fps |
| Median job end-to-end latency | 117.7 s | 129.7 s |
| Median server service time, including warmup/export/validation | 59.6 s | 74.6 s |
| Median queue time | 12.09 s | 0.024 s |
| Median worker inference time, including decode/export | 45.1 s | 55.8 s |
| Median upload time | 13.6 s | 15.1 s |
| Median result download and local verification | 28.9 s | 38.1 s |
| Sampled peak device memory, including loading and desktop use | 4.19 GiB | 7.57 GiB |
| Maximum per-worker PyTorch allocated memory | 3.06 GiB | 3.05 GiB |
| Busy capabilities request median / p95, 50 samples | 10.8 / 46.5 ms | 12.0 / 52.0 ms |

The initial idle request probe measured 8.4 ms median / 10.2 ms p95 over 50
requests. Normal timing workflows recorded no HTTP errors. Job p95 is deliberately
omitted: four jobs per configuration cannot establish tail latency.

**These are shared-load observations, not a controlled capacity benchmark.**
Other FPGA synthesis/simulation work continued. Host CPU activity had a median
of 99.7% / 99.6%; one-minute load ranged roughly 14–19 on 12 hardware threads.
Both configurations reached 88°C and reported thermal slowdown in 36 sampled
intervals. VRAM is a one-second sampled device total, not a guaranteed upper bound.
The route is local-network SSH, not a geographic/WAN test.

## Exactness

- All three fixtures passed with one and two workers: **432 raw prediction frames
  and 864 EXR files**, zero changed hashes against unchanged direct inference.
- All eight timed jobs passed: **1,152 downloaded EXR files** matched direct
  inference exactly. Local checksums were verified before publishing cache manifests.
- All 56 inference-source hashes match the previously accepted 602-frame suite.
  Neural network and renderer source files were not changed for serving.
- Live lifecycle checks passed: resumed uploads/downloads, owner isolation,
  idempotency conflicts, queued/running cancellation, retry after model reload,
  deletion and offline cache reuse. Running cancellation took 10.8 s. The retry's
  144 EXRs and a preview's 24 EXRs matched full direct inference.
- Fourteen HTTP/client regression tests pass, including rejected-request recovery,
  three-user queue fairness, invalid media, expiry and input limits.
- Scope is the prepared canonical RGB8 footage and fixed native profile on this
  execution stack. This does not qualify original camera-log/HDR decoding or
  establish matting accuracy against ground-truth mattes.

See [machine-readable measurements](pilot-results.json), the
[deployment guide](pilot.md), and [reproduction clients](../../benchmarks/api/README.md).
Media, credentials, individual request traces and direct-reference outputs remain
in private benchmark storage.
