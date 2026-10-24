# Editor inputs and corrections

The private v1 API accepts RGB plates, optional model hints and independent
coverage constraints. Discover enabled formats, profiles and limits with
`GET /v1/capabilities`. The [OpenAPI document](openapi.json) is the wire reference.
See [qualification results](editor-input-results.md) for the tested combinations.

## Plates and coverage assets

| Asset | Encoding | Interpretation |
|---|---|---|
| `image_sequence` | RGB8 PNG | Full-range sRGB transfer, BT.709 primaries, D65; no alpha |
| `image_sequence` | RGB half/float EXR | Scene-linear sRGB/BT.709, D65; finite negative and HDR values preserved |
| `matte_sequence` | Single-channel PNG, 8 or 16 bit | Coverage in [0,1], without a colour transform |
| `matte_sequence` | Single `Y` channel EXR, half or float | Finite coverage in [0,1] |

EXR inputs must be single-part scanline files, origin-zero matching data/display
windows, unit channel sampling and lossless NONE, RLE, ZIPS or ZIP compression.
RGB EXRs contain exactly R/G/B; coverage EXRs contain exactly Y. Mixed channel
types, tiled/multipart files, extra channels and nonfinite samples are rejected.
If RGB chromaticities are present, they must agree with the declared primaries.
Without that attribute, the explicit asset declaration remains authoritative.

Adapters convert camera log, ACEScg and other source spaces before upload and
record the conversion in `colour.transform_id`. A linear flag does not perform
a primaries conversion. VFR, interlacing, rotation, non-square pixels, input alpha
and arbitrary compressed video remain unsupported. Output is separate float32
ZIP EXR: straight linear RGB foreground and single-Y coverage alpha. An adapter
may premultiply locally after preserving the straight master.

## Frame alignment

All `frame_index` values are zero-based indices into the complete uploaded plate
context. Hints and masks must match its width, height, geometry, context
`frame_count`, rational frame rate, time base, start PTS and duration ticks exactly.
Timecode is descriptive and does not identify a frame. No resizing or retiming
is performed. Upload order and filenames have no temporal meaning.

The plate contains every frame. A coverage asset declares the plate's context
length even when its `files` list contains only one or a few frames. A correction
for context frame 12 stays frame 12 when the requested output starts at frame 24.
Preview and full jobs process the same complete context; output selection happens
after inference. Uploading a different context can change predictions.

## Model hints and carry

Add these optional fields to a normal job request:

```json
{
  "hint": {
    "asset_id": "asset_REPLACE",
    "mode": "keyframes",
    "confidence": 0.85,
    "frame_confidence": [{"frame_index": 12, "confidence": 0.35}]
  },
  "carry": {"enabled": true, "confidence": 0.6, "precedence": "provided"}
}
```

| Hint mode | Required coverage files | Behaviour |
|---|---|---|
| `first_frame` | Frame 0 only | Existing initial-matte conditioning |
| `sequence` | Every context frame | Existing dense matte conditioning |
| `keyframes` | One or more exact context indices | Sparse full-frame corrections |

Confidence is a model conditioning value in [0,1], not an opacity blend or an
accuracy guarantee. It defaults to 0.95. Only keyframes accept per-frame overrides;
each override must identify an uploaded correction. Sparse corrections are complete
mattes, not partial brush strokes with an unknown region. Missing keyframes supply
no observation; they are neither black mattes nor interpolated or held corrections.

Carry reuses predictions between overlapping Hann windows. It defaults to enabled,
confidence 0.95. With dense hints, `precedence=provided` keeps the supplied observation
where carry overlaps it; `carried` preserves the native runtime's legacy overlap
behaviour. The API defaults to `provided`. Explicit sparse keyframes always win
over carried predictions on their exact frames, with either precedence setting.
Carry can still affect uncorrected frames. Disable carry explicitly when desired.
The native CLI's defaults remain unchanged.

## Garbage, holdout and inside masks

Masks are independent coverage assets, applied to the final alpha after inference.
Specify the effect explicitly because editor terminology differs:

```json
{
  "masks": [
    {"asset_id": "asset_REPLACE", "mode": "static", "role": "force_background", "label": "garbage"},
    {"asset_id": "asset_REPLACE", "mode": "sequence", "role": "force_background", "label": "holdout"},
    {"asset_id": "asset_REPLACE", "mode": "static", "role": "force_foreground", "label": "inside"}
  ]
}
```

`static` contains frame 0 and repeats it across the context. `sequence` contains
every frame. Up to four masks are accepted. Let `K` be the maximum coverage across
force-foreground masks and `B` the maximum across force-background masks:

```text
final_alpha = min(max(model_alpha, K), 1 - B)
```

Background exclusion wins overlaps, including over hints and inside masks. Soft
values set bounds rather than multiplying opacity. Masks do not feed back into
temporal carry, change foreground RGB, synthesize missing foreground detail or add
despill. Raising alpha can therefore reveal imperfect predicted RGB. Labels are
descriptive only; separate garbage and holdout assets can share a role or use
different roles according to the adapter's intended effect.

## Cache identity and lifecycle

The effective recipe records content digests, exact hint indices/confidences,
carry settings, mask roles/modes, plate metadata, context/output ranges, output
settings and the qualified execution/model revisions. Its canonical SHA-256 is
`processing_fingerprint`, returned in result metadata and used as the CLI cache
key. Defaults are materialized. Asset IDs, labels, worker count, upload order and
temporary URLs are not pixel identities. Changing a hint file, timing, confidence,
mask, role or effective setting changes the fingerprint. Reordering equivalent
masks does not change it.

Capabilities retain `runtime_commit` as the accepted optimisation base and expose
`runtime_source_sha256` for the actual deployed Python bundle. The execution
fingerprint additionally binds worker/media/recipe code, dependencies and weights;
normal provisioning also requires the qualified GPU class and driver.

The client verifies input hashes and the returned recipe before publishing its
local cache. Resuming with changed guidance is rejected; create a new job/cache
directory for an edit. Existing verified local results remain available offline.
Guidance assets have the same ownership checks, expiry and job pinning as plates.
Deleting a hint or mask also cancels and deletes its dependent jobs/results.

## CLI guidance file

The shared client uploads local fixtures and replaces their paths with asset IDs.
Supply `--guidance /private/guidance.json` to `client.py ... run`:

```json
{
  "hint": {"fixture": "/private/corrections/fixture.json", "mode": "keyframes", "confidence": 0.85},
  "carry": {"enabled": true, "confidence": 0.6, "precedence": "provided"},
  "masks": [{"fixture": "/private/garbage/fixture.json", "mode": "static", "role": "force_background"}]
}
```

Paths are local client inputs and never enter HTTP requests. Each fixture uses the
same asset manifest as the API and a local `frame_dir`; PNG/EXR file slots use their
declared context indices. The server accepts only profiles advertised as enabled.
The client's `--allow-testing-profile` override is for isolated private qualification;
normal deployments reject unqualified execution builds during provisioning.
