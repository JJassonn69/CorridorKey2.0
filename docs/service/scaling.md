# Concurrent requests and inference engines

**Recommendation:** retain the optimised PyTorch worker, expose a worker-pool
limit, and qualify two independent workers on the existing GPU. vLLM and llama.cpp
are inference engines rather than substitute matting models. Porting CorridorKey
into either engine is a substantial alternative implementation, with no measured
speed or exactness benefit here.

## What their techniques would contribute

| Option | Potentially useful ideas | Fit for this pipeline |
|---|---|---|
| vLLM | Persistent models, admission control, continuous batching, caching, specialised kernels | Its model integration and attention interfaces would require adaptation; vision/language support does not automatically support dense video matting |
| llama.cpp / ggml | Portable kernels, persistent serving, CPU/GPU placement and bounded parallel sessions | Requires graph/operator/weight integration. Quantised execution changes arithmetic and is outside the current exact-output contract |
| NVIDIA Triton Inference Server | General model serving, multiple instances, queues and batching | More relevant as a future worker host; still needs our media/job/storage API and a state-aware pipeline wrapper |
| Existing PyTorch + Triton kernel | Already qualified operators, precision and model assets | Lowest-risk starting point; improve serving around it, then measure concurrent execution |

vLLM's [prefix cache](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
reuses language-model KV state for shared prefixes. CorridorKey's workload is
RVM plus frozen image features, temporal window inference and full-resolution
reconstruction, not token-by-token decoding. Its existing per-frame feature reuse
is useful, but cannot simply become an LLM prefix cache. vLLM also supports
pooling/multimodal models; the [supported-model list](https://docs.vllm.ai/en/latest/models/supported_models/)
and [integration guide](https://docs.vllm.ai/en/latest/contributing/model/basic/)
do not provide a drop-in CorridorKey runtime. This fit assessment is an inference
from those interfaces and the inspected architecture, not a port benchmark.

llama.cpp offers [parallel decoding and continuous batching](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
and [quantisation/hybrid CPU-GPU execution](https://github.com/ggml-org/llama.cpp).
Those mechanisms do not supply our temporal matting operators. Even an unquantised
port could change reductions, interpolation, normalization or output rounding.
Keeping identical weights is not enough to establish identical output.

[NVIDIA Triton's dynamic batcher](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
is intended for stateless model requests. The full CorridorKey job has temporal
state and ordered phases, so start with complete serial jobs per instance.
This server product is distinct from the Triton compiler used by our 7×7 kernel.
It may become worthwhile with multiple GPUs/backends, but adds no automatic speedup.

## What 22 GiB allows us to investigate

Concurrent HTTP sessions and simultaneous CUDA execution are separate limits.
The latest 720p case used about 3.48 GiB allocated / 3.71 GiB reserved; the 1080p
quality case used about 7.20 / 8.66 GiB. These are single-process measurements,
not safe per-worker reservations. Account for independent model copies, loading
spikes, CUDA contexts, workspace/allocator headroom, other GPU users, host RAM,
disk and CPU compression. Memory fit does not demonstrate compute headroom.

The 720p baseline delivers about 3.44 total frames/s. Two jobs that each run at
1.72 frames/s would provide the same combined throughput, with longer individual
service times. That is arithmetic, not a prediction. Measure completed frames
divided by the entire batch makespan, including drains; do not add isolated FPS.

Build a pool that can start one or two long-lived processes. Each owns its model,
scratch directory and cancellation channel. This avoids concurrent mutation and
CPU/GPU migration of a shared model. Serialise cold loads; admit jobs only if the
**measured combination** fits with a provisional 2 GiB whole-device reserve.
That reserve is a pilot safety policy to validate, not a proven sufficiency bound.
Keep one active job per user initially; different users can occupy the two workers.

## Qualification on the 22 GiB GPU

| Test | Measure / gate |
|---|---|
| One persistent worker, two fixed 720p jobs serially | Establish warm service time, total makespan, device/RAM peaks and CPU/disk load |
| Two persistent workers, same two jobs concurrently | Compare total frames/s, per-job service latency, p50/p95 response time, device peaks and exact output hashes |
| Bursts of 1/2/4/8 requests from multiple users | Queue wait separately from service time; fairness, rejection/backpressure and cancellation responsiveness |
| Two 1080p jobs; mixed 720p + 1080p | Separate capacity profiles after uncontended 1080p baseline; no extrapolation from 720p |
| Cold load, warm repeat, shape changes and repeated cancellations | Loading headroom, leaks, orphaned children, worker restart and A→B→A output equality |
| Long enough sustained runs | Thermals/clocks, repeated measurements, host memory/disk and upload/download interference |

Compare the same job set, decode/output settings and seeds, interleave test order,
record background CPU/GPU activity, and retain the original raw and EXR exactness
gate. Use enough repetitions to estimate latency distributions; two runs cannot
support a reliable p95. Compare each job to its serial reference, including the
full 602-frame suite across every newly enabled profile combination.

Proposed promotion rule: reproducible aggregate throughput gain (target at least
10%), exact outputs, stable memory below admission limits, and acceptable measured
tail latency/fairness. The 10% threshold is a policy proposal; latency targets must
follow the pilot workload. If two workers only divide the same compute capacity,
keep one active worker and accept concurrent queued requests.

## Scaling sequence

1. Persistent loading, fair queue, client caching, then the two-worker experiment.
2. Tune CPU/export threads and storage for the selected worker count. The current
   twelve EXR writer threads per alpha/FG job can oversubscribe a six-core host.
3. Reuse completed results within the same owner and exact recipe. Share an in-flight
   computation only after defining subscriber cancellation and retention semantics;
   one client cancelling must not invalidate another client's accepted job.
4. Investigate same-shape Stage 2 batching or separate per-job CUDA streams only if
   profiles show idle capacity. Dynamic batching changes GEMM dimensions and may
   change rounding; the existing tail-batch preservation is evidence of that risk.
   Shared immutable weights need a worker redesign because current phase offloading
   mutates module placement. Preserve each clip's temporal state and window plan.
5. Add a second GPU as independent job capacity when demand warrants it. Keep the
   public API unchanged; one scheduler can route local GPU workers. Use per-GPU
   qualification and model copies before considering tensor/model parallelism.
6. Consider a general inference server or database/object-store migration only
   when measured load or a second host requires it. A vLLM/ggml rewrite should have
   its own exactness and end-to-end performance case before replacing this worker.

No sponsor status enters scheduling, quality selection, quotas or cache access.
