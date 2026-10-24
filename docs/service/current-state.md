# Inspected inference baseline

Inspected 2026-10-24: application commit `86df1a18bcfabc72296db6764ca1449aecf8011f`,
based on upstream `485f5e9ab759ead6138cc990776184b59d6ea971`. All 56 Python source
hashes match the latest acceptance manifest. This is an inspection of code and
existing evidence, not a new performance run. See the later [API pilot results](pilot-results.md).
This is the original optimisation snapshot. The later [editor-input revision](editor-inputs.md)
adds optional hint branches and service inputs; its [qualification](editor-input-results.md)
records source identity and reruns the frozen baseline.

## Entry points and execution

| Implemented | Service implication |
|---|---|
| `infer.py`: load one model, synchronously process one video or frame directory | Original direct entry point; the separate API wrapper now supplies the job lifecycle |
| `run_interactive.py`: guided launcher; subprocess per clip | Batch launching does not provide persistent model reuse |
| `corridorkey.runtime.load_model()` / `run_inference()` | Useful worker boundary; not a thread-safe serving interface |
| `RuntimePaths`, `InferenceConfig`, `WriterConfig` | Internal configuration, not a public API contract |
| Native CUDA BF16 inference; optional version-guarded Triton depthwise kernel | Pin the execution stack and disclose a distinct execution revision |

Retained optimisations: per-frame foundation reuse/offloading, selective BF16
parameter storage, unused-head removal, bounded input prefetch, parallel EXR
writing, exact byte-value conversion lookup, and ordered 7×7 depthwise convolution.
CSwin parameters retain their original precision. No replacement SAM/depth model
was accepted: the separate helper-model study found regressions.

## Measured scope

Environment: RTX A5000 reporting 23,028 MiB (about 22.49 GiB), Ryzen 5 2600
(6 cores / 12 threads), Python 3.12.13, PyTorch 2.9.1+cu128, Triton 3.5.1,
NumPy 1.26.4, OpenCV 4.11.0.86, driver 580.126.09.

| Evidence | Observation | Limit |
|---|---|---|
| Latest 720p, 72 frames | Median 20.906 s = 3.44 frames/s; runs 21.025 / 20.786 s | Native decode through float EXR alpha/FG export, after eight warmup frames |
| Latest 720p VRAM | Max allocated 3.482 GiB, reserved 3.707 GiB; sampled whole-device peak about 4.31 GiB | Allocator peaks exclude load/warmup; device sampling can miss spikes |
| Latest 720p model load | 10.680 / 10.692 s | Reuse could avoid repeated loads; sustained reuse is not yet qualified |
| Latest 1920×1088 quality capture | Allocated 7.205 GiB; reserved 8.658 GiB; padded to 1920×1152 | Instrumented quality run, not a throughput measurement |
| Prior 1080p speed round | 58.122 → 50.073 s before the depthwise change | Thermal slowdown; latest 1080p throughput remains unqualified |
| Exactness | 602 frames / 11 configurations; 1,806 raw/EXR files byte-identical to their references | Preservation on tested inputs/stack, not a universal quality guarantee |
| Tests | 42 experiment tests; 17 focused tests passed after packaging | No multi-user service, long-session, or GPU concurrency tests |

The latest timed profile fixes Hann 24/12, Stage 2 batch 1, scale 1.0,
`auto_vram=False`, normal mode, carry hints on, and float EXR alpha/FG output.
It is not interchangeable with the CLI's automatically selected profile.
The final 720p timings had no sampled thermal slowdown or synthesis overlap;
other CPU background work remained. Latest 1080p timing was excluded for
contention and thermal slowdown. Do not combine speed ratios across rounds.

The README's 4 GB / 6 GB guidance and the runtime's automatic VRAM lookup table
are not new capacity measurements for this optimised build. That table still
contains original values such as 7.52 GiB for 720p/24. No measured service
capacity, 4K throughput, network budget, host-RAM peak, or concurrent-worker
memory budget exists. Avoid deriving worker count by dividing 22 GiB by 3.48.

## Media behaviour

| Area | Actual behaviour |
|---|---|
| Video input | OpenCV `VideoCapture`; frame count/FPS metadata, sequential BGR frames. No retained per-frame PTS, VFR contract, explicit gamut conversion or source-bit-depth guarantee |
| Image input | Sorted EXR/PNG/JPG/JPEG paths. Non-EXR reads use `IMREAD_COLOR` and divide by 255; this is not a guaranteed 16-bit PNG pipeline. EXR uses unchanged values, then float32; input alpha is discarded |
| Time/ranges | `num_frames`, and video `begin` / `middle` / `random_middle`. No arbitrary public `[start,end)` range; frame directories take a prefix. FPS is a float |
| Geometry | Reflect-pad to a multiple of 128, then crop to native dimensions. CLI advertises `model_scale`, but `_resize_output_torch()` rejects non-native scaling |
| Colour | Display inputs follow sRGB transfer functions; linear EXRs require `frame_dir_linear`. No OCIO/ACES/gamut-management API. Linear EXR flag does not identify primaries |
| EXR output | Alpha, raw foreground, straight RGBA cutout, checker. Half or float, ZIP. Float RGB preserves HDR/negative model values; alpha clamps to [0,1]. Half clamps to ±65504 and adds quantisation |
| Video output | Clipped display output; alpha/FG/checker use ffmpeg (default x264, 10-bit pixel format); cutout uses ProRes 4444 path. No audio output. Encoded video is not an exact archival matte |
| Alpha/foreground | `fg` is unassociated RGB. EXR `cutout` is straight RGBA with despill; raw `fg` is not despilled. Premultiplication is not an exposed output option |
| Hints | First-frame and full-sequence hints exist; strength, carry behaviour, windows and batching affect the reproducibility contract |

Fewer temporal frames, different Hann windows, cropped context, new batching,
resizing or different hints can change results. RVM state runs continuously over
the selected clip; Hann blending and carried hints also depend on boundaries.
A separately processed subclip need not equal the same frames from a full clip.
Float32 files do not turn BF16 model arithmetic into FP32 model accuracy.

## Concurrency and cancellation

- Module placement moves between CPU/GPU during a job. Low-memory configuration
  mutates module attributes and the normal path does not explicitly restore them.
  A→B→A, mixed-profile, cancelled-job and long-lived-worker reuse need testing.
- RVM state and temporal accumulators are local to a run, but that alone does
  not make the shared model safe for simultaneous calls.
- Default scratch names derive from the clip basename. Concurrent requests need
  unique owner/job/attempt directories, even when input filenames differ only by path.
- Six EXR threads per output means twelve writer threads for alpha+FG, plus
  input prefetch and tensor work on the six-core host. Buffers are bounded, but
  multiplying jobs also multiplies host work and buffering.
- Progress is printed, not emitted as structured events. No cancellation token,
  timeout, resumable inference checkpoint or retry protocol exists. The launcher
  catches `KeyboardInterrupt` but does not implement explicit child-group cleanup.
- Normal exceptions run cleanup and EXR pool draining. Forced termination cannot
  rely on `finally`; partial exports can remain. Video writer queue/pipe/join/wait
  operations lack deadlines, so cancellation cannot promise immediate shutdown.

## Evidence locations

Application references: [CLI](../../infer.py), [runtime](../../corridorkey/runtime.py),
[media I/O](../../corridorkey/video_io.py), [kernel](../../corridorkey/depthwise.py),
[packaged results](../performance.md), [third-party notices](../../THIRD_PARTY_NOTICES.md).

The separate local `corridorkey-bench` workspace retains
`results/speed3/acceptance.json`, `results/speed3/summary.json`, the per-run
`manifest.json` files, `bench/pr/verification.json`, and reports under
`bench/{optimization,speed,speed2,speed3,quality}/`. Full captures remain on the
test server. Those private workspace paths are evidence references, not service
API inputs or publicly downloadable fixtures.
