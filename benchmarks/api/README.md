# Mac client / GPU API latency tests

Status, 2026-10-24: private API deployed and tested from the Mac. See the
[deployment guide](../../docs/service/pilot.md) and [measured results](../../docs/service/pilot-results.md).

## Private footage fixtures

The user downloaded three free 2K masters from
[VFX Assets](https://vfxassets.com/d/demoreel-footage-for-compositors?mode=greenscreen):

| Clip | Selected source frames, zero-based `[start,end)` | Source rate | Test input |
|---|---|---|---|
| Woman in White Stands Holding Pistol (`escape-s001`) | 48–120 | 24000/1001 | 72 PNGs, 1024×578 |
| Saber Lock Struggle Closeup (`saber-s009`) | 4–76 | 24/1 | 72 PNGs, 1024×578 |
| Man Climbs Ladder, Wide (`ladder-s001`) | 24–96 | 24000/1001 | 72 PNGs, 1024×578 |

Usage is private personal testing only. Keep masters, extracts, mattes and visual
reviews outside Git and public artifact downloads; no training or redistribution.
The user's originals remain in Downloads. Prepared files and manifests are in the
separate benchmark workspace at `data/vfx-api/`; request metadata/digests are recorded
in [vfx-sources.json](vfx-sources.json). There are 216 frames / 227,025,169 PNG bytes.
These are distinct fixtures from the earlier 720p throughput benchmark.

The masters are 2048×1156 ProRes 4444, 12-bit YUV, with S-Log3/S-Gamut3.Cine
identified by the provider. Container tags say BT.709 and do not specify range;
the fixture decoder explicitly uses the BT.709 YUV matrix and limited-to-full
range conversion. This interpretation is recorded and is not a colour-critical
certification of the masters.

Preparation uses OCIO 2.6.0, built-in
`studio-config-v2.2.0_aces-v1.3_ocio-v2.4`, Sony `S-Log3 S-Gamut3.Cine` to
`Linear Rec.709 (sRGB)`, then float-channel Lanczos downsampling to exactly half
dimensions. Apply the sRGB transfer, clip to [0,1] and round to RGB8 PNG. No ACES
view/tone-mapping transform is baked in. The SDR derivatives clip out-of-gamut,
negative and over-range samples; they cannot qualify the original HDR/12-bit path.
Full values, versions, hashes and clipping counts are retained in each private
`fixture.json`. Preserve the masters for a separate linear-EXR test later.

Reproduction (the preparation script refuses to overwrite an existing clip):

```sh
python benchmarks/api/prepare_vfx.py --downloads /path/to/Downloads --output /private/fixture-root
```

Preparation needs `imageio-ffmpeg`, `opencolorio`, NumPy and Pillow in an isolated
environment; all installations must use SFW. The latency client below uses only
Python's standard library. Existing system ffmpeg was left untouched after its
missing-Theora-library failure; preparation used the imageio packaged binary.

## Deployment path and measurement definitions

Use this Mac as the actual uploader/downloader. For the private pilot, bind the
GPU API to server loopback and connect through the existing SSH/ProxyJump route.
Pass the resulting local forwarded origin as `--base-url`. SSH encrypts the network
transport; plain HTTP is accepted by the client only on loopback. Public HTTPS can
be measured as a separate route later. The Mac and GPU are different machines,
but this is not an independent geographic/WAN test. Record the route with results.
Use `--route-label` for that description. UTC timestamps and server request IDs
support log correlation; all elapsed-time calculations still use a monotonic clock.

| Metric | Definition |
|---|---|
| Request headers latency | Client monotonic time immediately before sending through receipt of response headers; includes connection setup when needed and request-body transfer |
| Complete request latency | Same start through complete response read / streamed local write; group by operation and payload size |
| End-to-end latency | Before local fixture verification and asset declaration through all returned files verified and the local cache manifest atomically published |
| Upload / validation / submission / wait / download | Client-observed stage intervals; job wait includes queue, service and polling detection |
| Queue / service time | Optional `server_timings` measured on one server monotonic clock; null if absent, never estimated by cross-host wall-clock subtraction |

Source downloading and colour preparation are excluded from service end-to-end
latency. Report their one-time costs separately if needed. A resumed workflow is
marked `resumed`; do not mix it with fresh full-workflow samples. HTTP connections
are reused and every attempt, byte count and error is retained. Percentiles are
per operation; p95 is null below 20 samples. Errors are included and counted.

## Run after the private API is deployed

Obtain disposable per-user service tokens from the deployment, store them with
mode 0600, and use fresh output directories for independent measurements. Never
put tokens on the command line or in Git. Select an advertised immutable profile:

```sh
python benchmarks/api/client.py --base-url http://127.0.0.1:18080 --token-file /private/token probe --count 50 --output /private/results/requests.json
python benchmarks/api/client.py --base-url http://127.0.0.1:18080 --token-file /private/token run --fixture /private/fixture-root/escape-s001/fixture.json --profile-id PROFILE --output /private/results/escape-full
python benchmarks/api/client.py --base-url http://127.0.0.1:18080 --token-file /private/token run --fixture /private/fixture-root/escape-s001/fixture.json --profile-id PROFILE --purpose preview --frames 24:36 --output /private/results/escape-preview
```

Resume an interrupted client by adding `--resume` with the identical options and
output directory. It reconciles uploads with HEAD, retains idempotency keys, resumes
downloads with ETag/If-Range, checks SHA-256 and rejects changed input/profile/range.
Output JSON contains request records, operation summaries, stage timings and the
server's optional timings. The client does not assume an API is healthy or fabricate
missing queue/GPU timings. Lifecycle/fairness/security checks below require the
real server; passing the client transport tests does not certify those features.

## Test matrix after deployment

| Test | Required check |
|---|---|
| Idle and busy request probes | At least 50 capability/status requests per condition; median/p95, errors, cold connection versus reused connection |
| Fresh full workflow, each clip | Local verify + upload + validation + queue + service + download + cache; repeat and report all samples, not only fastest |
| Preview of frames 24:36 | Match those exact frames from the full 72-frame-context result |
| Direct versus API | Run the same canonical PNGs, model/profile and export settings directly; require identical raw outputs/EXR hashes before accepting serving changes |
| Interrupted upload/download | Stop client mid-transfer; resume without duplicate bytes or corrupted cache |
| Duplicate submission / retry | Same key returns same job ID; changed payload conflicts; explicit retry gets a new job ID and retains recipe |
| Cancellation | Queued/running/draining races; no partial publication; next job still matches reference |
| Two test users, 1/2/4/8 request bursts | Ownership isolation, queue bounds, fairness, 429 behaviour; record queue wait separately |
| One versus two GPU workers | Same job set and output checks; total frames/s, per-job latency, loading/device/RAM peaks and thermals; follow the existing qualification gate |
| Expiry and deletion | Revoke access, wait for actual cleanup, confirm no leaked temporary files |
| Offline cache | Verified local playback/read works after disconnecting client and deleting remote results |

Pin model/window/precision/decoder/renderer settings, record server CPU/GPU load,
thermals, network route, bytes and worker count. Record every exclusion. Do not
pool tiny preview responses, PNG uploads and EXR downloads into one latency number.

## Client verification

```sh
python -m unittest discover -s tests -p test_api_benchmark_client.py -v
```

Seven real loopback HTTP transport tests cover preserved idempotency keys across
503 retries, lost PATCH replies, resumed downloads, checksum failures, credential
origin/redirect checks, sample-count handling and complete workflow/cache resume.
They use deliberately fake media bytes and no inference model. No test latency is
reported as deployed-API performance.
