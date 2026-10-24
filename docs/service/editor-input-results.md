# Editor-input qualification — 2026-10-24

**Accepted and enabled:** first-frame/dense/sparse hints, confidence and carry,
linear half/float EXRs, explicit coverage constraints and a 1080p profile.
No retraining or replacement neural model. The native inference defaults retain
all 602 frozen baseline frames exactly. [Machine-readable evidence](editor-input-results.json)
records the deployed source/worker fingerprint, cases and capacity measurements.

| Check | Result |
|---|---|
| Frozen baseline: native, tails, low-memory, linear EXR and legacy hints | 602 frames / 1,806 raw and rendered files identical |
| Eight input/guidance cases through the Mac API | 260 raw frames / 520 EXRs identical to independent direct calls |
| Two concurrent workers: sparse corrections and HDR plate with constraints | 72 raw frames / 144 EXRs identical to the direct references |
| 1920×1080, 144-frame context, first-frame hint | 144 raw frames / 288 EXRs identical to the accepted direct runtime |
| Small job after the large job on the same worker | 36 raw frames / 72 EXRs identical; no state leakage |
| Exclusive 1080p admission with two resident workers | Second user's job queued 205.1 s; service intervals did not overlap |
| Cache safety | Changed guidance rejected before HTTP; identical replay reused the job and all 72 local files |
| Contract tests | 21 passing on Mac and server; two additional CPU tests inspect actual hint tensors across overlapping windows |
| Normal deployed API, raw capture disabled | HDR + sparse corrections + masks: all 72 downloaded EXRs and the processing fingerprint match the qualified reference; capability/job/result schemas pass |

New feature comparisons total **512 raw frames and 1,024 EXRs**. The original
runtime is the reference for existing first/dense-carried hints and the large/half
cases. Sparse corrections and provided-dense precedence use independent calls to
the extended runtime. Constraints additionally use an independent piecewise mask
oracle. Every reference and published export is physically hashed. The Mac checks
downloaded bytes and exact frame/PTS/colour metadata before publishing its cache.

The eight cases cover first-frame hints, dense hints under both precedence modes,
sparse confidence overrides with carry enabled/disabled, overlapping static and
animated constraints, linear float32 HDR with sparse corrections/constraints, and
half EXR with a first-frame hint. The 1024×578 HDR plate contains values from
−0.07778754 to 2.13053226. Prior predictions used as guidance are not ground truth;
these checks establish serving equivalence, not a measured visual improvement.

## Memory and admission

RTX A5000, 23,028 MiB, driver 580.126.09. Approximately one-second device samples
are restricted to API service intervals; independent reference runs are excluded.
Device totals include desktop use and any idle resident model.

| Workload | Sampled peak device memory | Maximum worker allocated / reserved |
|---|---:|---:|
| One worker, eight new feature cases | 4,292 MiB (4.19 GiB) | 3.06 / 3.58 GiB |
| Two workers, concurrent sparse/HDR requests | 6,817 MiB (6.66 GiB) | 3.06 / 3.59 GiB |
| One 1080p job with two models resident | 11,117 MiB (10.86 GiB) | 7.20 / 8.66 GiB |

Keep **one worker by default**. Two 720p jobs may execute concurrently for different
users; 1080p jobs require an otherwise idle GPU queue even with two workers.
720p is limited to 1280×720 / 72 context frames; 1080p to 1920×1080 / 144.
Larger dimensions, longer contexts and concurrent 1080p jobs remain unqualified.
The large capacity case used RGB8; the HDR cases used 1024×578 plates. This is a
tested matrix, not every possible combination of format, size and guidance.

These are correctness/capacity runs with raw capture enabled, not speed benchmarks.
Other CPU work continued. The 1080p run reached 88°C and recorded thermal slowdown
in 28 device samples; no clean throughput claim follows. Sampled device memory
is not a guaranteed upper bound. See the historical [pilot timing results](pilot-results.md)
for the existing default-worker decision.

## Reproduction and scope

The [benchmark tools](../../benchmarks/api/README.md#editor-input-qualification)
prepare private fixtures, submit Mac workflows, independently compare direct
inference and extract capacity samples. Input media, tokens, full manifests and
logs remain in the private benchmark workspace/server. No footage is committed,
redistributed or used for training. Timings retain the actual SSH/ProxyJump route
and shared-load conditions.

Normal provisioning accepts only the source/worker/weight fingerprint and GPU/driver
in [qualification.json](../../service/qualification.json). Tests ran on a separate
port/data directory with profiles marked `testing`. Normal clients reject testing
profiles unless explicitly overridden. See [editor inputs](editor-inputs.md) for
formats, precedence and cache semantics, and [deployment](pilot.md) for private access.
