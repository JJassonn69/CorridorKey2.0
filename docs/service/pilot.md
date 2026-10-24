# Run the private API pilot

The pilot serves the v1 contract on `127.0.0.1:18080`. Connect through SSH from
the client. It wraps the accepted inference implementation without changing
its neural network or rendering code. Results and credentials stay private.

## Prepare the server

Use the existing, pinned Linux/CUDA inference environment and downloaded model
assets. Install the control-plane validator through the package firewall:

```sh
sfw uv pip install --python .venv/bin/python -r service/requirements.txt
.venv/bin/python -m service.admin init \
  --package /absolute/model-package \
  --data /absolute/private-data \
  --config /absolute/private-data/config.json --workers 1
.venv/bin/python -m service.admin token \
  --data /absolute/private-data --owner test-user \
  --output /absolute/private-data/test-token
.venv/bin/python -m service.server --config /absolute/private-data/config.json
```

Run the server under a user service with `KillMode=control-group` so a controller
failure also terminates its CUDA processes. Keep the data directory outside the
repository. The server creates owner-only files; tokens expire after 72 hours by
default and only their SHA-256 digests are stored in SQLite. Revoke a token locally
with `.venv/bin/python -m service.admin revoke --data /absolute/private-data --token-file /private/token`.

`init` fingerprints runtime sources, checkpoint, all foundation assets, execution
stack, worker wrapper, and fixed profile. It changes configuration, not running
workers. Stop/restart the managed service to change one versus two workers.
Never start an independent CUDA pool against the same data directory.
Provisioning rejects a different execution fingerprint, GPU class or driver from
the checked-in qualification record. Requalify changed inference environments
before updating that record; do not simply copy its fingerprint to bypass the gate.

## Connect and process footage

Forward local port 18080 to the server's loopback port through authenticated SSH.
Copy your token into a private local file with mode `0600`; do not put it in URLs,
shell arguments, Git, or logs. Then use the [Mac client](../../benchmarks/api/README.md).

```sh
python3 benchmarks/api/client.py \
  --base-url http://127.0.0.1:18080 --token-file /private/client-token \
  probe --count 50 --output /private/idle-probe.json
python3 benchmarks/api/client.py \
  --base-url http://127.0.0.1:18080 --token-file /private/client-token \
  run --fixture /private/fixture.json --profile-id native-720p-24x12 \
  --output /private/new-run
```

Use `--resume` with the same output directory after an interrupted workflow.
The local manifest is published only after every result passes its SHA-256 check.
Playback can use those local files without a server connection.

## Implemented limits

| Concern | Pilot behavior |
|---|---|
| Input | Contiguous RGB8 PNG sequence, sRGB/BT.709, progressive square pixels, no alpha |
| Size | Up to 72 frames, 1280×720; the measured footage is 1024×578 |
| Profile | Hann 24/12, Stage 2 batch 1, native scale, seed 0, normal memory mode |
| Output | Float32 ZIP EXR, separate straight linear foreground and coverage matte |
| Alpha channel | Native single EXR channel `Y`, interpreted as coverage; no file rewrite |
| Preview | Process all uploaded context, deliver the requested frame range |
| Workers | Independent persistent model copies; one active job per worker and per user |
| Queue | Eight pending globally, two pending per user, round-robin users/FIFO per user |
| Transfer | Durable offsets, whole-file SHA-256, frozen validation, authenticated byte ranges/ETags |
| Cancellation | Queued jobs cancel immediately; running jobs cannot publish after cancellation and are killed/reaped after up to 10 s grace; the model reloads |
| Recovery | Unknown worker/CUDA failures recycle the process; controller restart fails interrupted attempts |
| Storage | 1 GiB input asset, 4 GiB user reservation, 10 GiB server disk floor |
| Expiry | Uploads 24 h from creation; ready inputs 24 h from validation; results 24 h from completion; accepted jobs pin inputs |
| Deadlines | 600 s queue age and 600 s execution; explicit retry creates a new job |
| Access | Expiring owner tokens; cross-user resource IDs return 404; sponsorship changes no processing policy |

Every job uses a fresh configuration and private scratch directory. Each job has
the direct benchmark's eight-frame warmup; its cost is included in API service
latency. Model loading happens before the worker becomes ready. Capabilities
report ready/busy/starting workers and whether raw qualification capture is on.

Worker inference time includes decode, model execution and EXR export. Service
time also includes warmup and final validation/publication. Queue time uses the
server's monotonic clock. Mac end-to-end time additionally includes local input
verification, upload, polling, download and local checksum validation.

## Qualify changes

`--capture-raw` on `service.admin init` enables private raw-prediction digest
capture. Run [qualify.py](../../benchmarks/api/qualify.py) against the unchanged
direct benchmark for each fixture. Timing runs disable capture and still compare
every downloaded EXR with the direct reference. Never mix qualification timings
with performance measurements.

[pilot.py](../../benchmarks/api/pilot.py) submits two independent client lanes,
records busy request probes, and checks local outputs against references before
releasing remote storage. [lifecycle.py](../../benchmarks/api/lifecycle.py) checks
real upload/download resume, ownership, cancellation/reload, retry, preview,
deletion and offline caching. Shortened expiry and invalid media are tested by
the controller/HTTP tests; the real test does not wait 24 hours for retention.

## Deployment boundary

This is a private loopback pilot using a bounded Python HTTP server, SQLite WAL,
one PNG validator thread and spawned CUDA processes. It has no public ingress,
editor plugin, account signup, billing, arbitrary codec upload or distributed
queue. A public deployment still needs hardened HTTPS ingress/request-rate policy
and a separate restricted decoder process. Do not expose the pilot port publicly.

See [measured results](pilot-results.md) for the worker decision and the limits of
the shared-load comparison.
