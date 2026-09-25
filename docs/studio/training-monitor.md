# Training Monitor

The Training Monitor provides live SNN training from the Studio web IDE.
Configure network architecture, surrogate gradients, and training
hyperparameters, then watch loss curves, accuracy, and per-layer spike
rates update in real time via Server-Sent Events.

## Quick Start

1. Switch to the **Train** tab
2. Select dataset (Synthetic for fast demo, MNIST for real training)
3. Choose surrogate gradient function
4. Set epochs, batch size, learning rate, timesteps
5. Optionally enable learnable beta and threshold
6. Click **Train** — charts update live as epochs complete
7. Export a checkpoint JSON when a job ID exists, or import a previous
   checkpoint to restore its training configuration
8. Click **Stop** to abort early

## Features

### Live Metric Streaming

Training metrics stream from backend to frontend via SSE (Server-Sent
Events).

The browser reads this same-origin stream with authenticated `fetch`, using
the Studio bearer header. The token is never placed in the stream URL and
redirects are refused. Closing the monitor stream aborts the pending read.
On opening the Training Monitor, the browser loads retained training runs and
observes the newest one. Selecting another run fetches its current status and
replays its persisted metric stream. If the stream disconnects, `Refresh runs`
rechecks the selected run and reconnects its stream. A retained run's
configuration is shown separately from the editable project settings. Rows
admitted before the configuration snapshot say `config not recorded`; checkpoint
export is unavailable for those rows. Runs with a retained configuration can
export a portable checkpoint after API restart. Reattachment does not claim a
new guided-flow training result for the currently edited project: the retained
run's seed and clipping settings are not editable project fields.
An `unknown` or disconnected run remains an active, uncertain observation in
the panel. Refresh first: Stop is disabled until the API confirms a live run,
and the panel does not offer a new Train action under that selection. A direct
Stop request against a durable `unknown` record returns `unknown` without
claiming cancellation.

Each epoch emits:

- **train_loss**, **val_loss** — spike count cross-entropy
- **train_accuracy**, **val_accuracy** — classification accuracy
- **layer_spike_rates** — mean firing rate per spiking layer
- **param_snapshot** — current beta and threshold values (if learnable)

### Configurable Architecture

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Dataset | synthetic | synthetic, mnist | Input data source |
| Epochs | 10 | 1–100 | Training epochs |
| Batch Size | 64 | 8–512 | Mini-batch size |
| Learning Rate | 0.001 | 0.0001–0.1 | Adam optimizer LR |
| Timesteps | 25 | 5–100 | SNN temporal unrolling steps |
| Hidden | [128] | — | Hidden layer sizes |
| Surrogate | atan | 6 options | Surrogate gradient function |
| Learn Beta | off | on/off | Learnable membrane leak |
| Learn Threshold | off | on/off | Learnable spike threshold |

### Surrogate Gradient Functions

Six surrogate gradient approximations for the non-differentiable
Heaviside spike function:

| Function | Reference | Formula (backward) |
|----------|-----------|-------------------|
| atan_surrogate | Fang et al. 2021 | alpha / (2(1 + (pi*alpha*x/2)^2)) |
| fast_sigmoid | Zenke & Vogels 2021 | slope / (1 + slope*|x|)^2 |
| superspike | Zenke & Ganguli 2018 | 1 / (1 + beta*|x|)^2 |
| sigmoid_surrogate | standard | sigmoid'(slope*x) |
| straight_through | Bengio et al. 2013 | identity |
| triangular | Esser et al. 2016 | max(0, 1 - |x|/width) / width |

### Neuron Cell Types

The training backend uses sc-neurocore's 11 PyTorch-compatible spiking
neuron models:

- **LIFCell** — leaky integrate-and-fire (default)
- **IFCell** — integrate-and-fire (no leak)
- **ALIFCell** — adaptive LIF (Bellec et al. 2020)
- **ExpIFCell** — exponential IF (Fourcaud-Trocme et al. 2003)
- **AdExCell** — adaptive exponential IF (Brette & Gerstner 2005)
- **LapicqueCell** — retained SC hard-reset RC training cell (historical name)
- **AlphaCell** — alpha synaptic conductance
- **SecondOrderLIFCell** — LIF with second-order dynamics
- **RecurrentLIFCell** — LIF with within-layer recurrence
- **ConvSpikingNet** — convolutional SNN (2D spatiotemporal)
- **SpikingNet** — multi-layer feedforward SNN

### Visualisations

The monitor displays four live panels:

1. **Loss Curve** — train loss (blue) and val loss (red) per epoch
2. **Accuracy Curve** — train accuracy (green) and val accuracy (purple)
3. **Layer Spike Rates** — horizontal bar per spiking layer, showing
   mean firing rate as percentage
4. **Parameter Evolution** — current values of learnable beta and
   threshold parameters

### Job Lifecycle

Training starts through the Studio job manager. In the web backend,
`/api/training/start` uses an isolated process-backed worker task; direct
module callers can still use the legacy in-process thread path for local
compatibility. The lifecycle is:

```
idle → starting → running → completed | stopped | failed | interrupted | unknown
```

Multiple training jobs can run concurrently. Each job has a unique ID
used for status queries and SSE stream subscription.
Stop is cooperative: the worker checks it before each training batch, between
training and validation, during validation, and before reporting completion.
An already running tensor operation finishes before the next check.
If a run reaches a terminal state before Stop arrives, the Stop response reports
that state in the Training Monitor vocabulary. In particular, a cancelled
platform job is `stopped`; the browser does not replace a completed outcome with
`stopping` when a delayed Stop response arrives.
After restart, `interrupted` means the ledger proved the supervisor is gone;
`unknown` means liveness could not be established. Neither is presented as
completed. Historical jobs admitted before the configuration snapshot was
introduced remain listed with `config: null`.
The SSE endpoint tails a retained training job's bounded event log through the
manager even when the serving API process has no local proxy. Active and
`unknown` records keep the stream open with heartbeats until their durable
status resolves. Terminal records yield their recorded events followed by one
terminal outcome when the event log has no terminal frame. The training status
and stream routes refuse records of another Studio job kind.
An interrupted record emits an `interrupted` SSE event; the browser keeps that
status distinct from a training failure and closes the stream. A failed
computation still emits `error`.

### Service Responsibility Boundary

`sc_neurocore.studio.training` remains the historical public facade. Its
implementation is separated into bounded, one-way responsibilities:

- `_training_job` owns PyTorch discovery, the training loop, checkpoint
  publication, and worker-side live-attach application.
- `_training_datasets` owns dataset construction and seeds the Python, NumPy,
  and Torch generators before data or model creation.
- `_training_control` owns the parent-process registry, status reconciliation,
  checkpoint import/export, and cancellation.
- `_training_stream` owns SSE framing, bounded event tailing, and durable job
  observation after the API process loses its local training proxy.
- `_training_attach` owns warm-start and live-attach orchestration across
  verified job artifacts and confined worker channels.
- `_training_events` owns portable JSON values, event persistence, and
  platform-status event translation.
- `platform.training_process` remains the importable process-task entry point.

The facade depends on these modules; implementation modules do not import the
facade. Architecture tests enforce that dependency direction, stable public
exports and signatures, pickle identity, file-size ceilings, and all 12
composed HTTP routes.

The Training Monitor execution backend is Python with PyTorch. No callable
Rust, Julia, Go, or Mojo counterpart is wired to this service contract, so the
responsibility split makes no cross-language parity or throughput claim. A
seeded real-Torch regression pins the established metrics, architecture,
parameter count, and learned tensor-state digest while the algorithm remains
unchanged.

Completed jobs expose a Training Monitor evidence summary for
`training/evidence.json`, result artifacts, replay route, and terminal status
without host-local paths. The summary validates `training` evidence
classification and terminal statuses through the shared Studio
evidence-classification contract; malformed or non-terminal evidence artifacts
return a bounded unavailable summary instead of untrusted metadata.

### Portable Checkpoints

Training checkpoints use the `studio.training.checkpoint.v1` schema. They are
portable JSON manifests for Studio configuration, terminal status, metrics, and
evidence metadata. They do not expose local filesystem paths and do not include
raw model-weight tensors. Export and import validate embedded
`studio.training.evidence-summary.v1` metadata as verified `training` evidence
with terminal status, replay route, payload digest, and path-free artifact
manifests. Import also validates both the config digest and full checkpoint
digest before returning the restored training config to the UI. The browser
import control first validates the JSON schema, lowercase SHA-256 digest
fields, training config shape, and optional weight-artifact paths before
submitting the checkpoint to the backend.

Completed process-backed training jobs also publish binary model weights as
job artifacts:

- `training/model_state.pt` stores a PyTorch `state_dict` payload with the
  training config, model info, final metrics, and learned parameters.
- `training/model_state.json` stores path-free metadata using
  `studio.training.weight-checkpoint.v1`, including artifact size, SHA-256,
  framework, format, architecture, parameter count, config digest, and final
  metrics.

The portable checkpoint JSON may include that weight metadata under
`weight_checkpoint`, but the raw tensor payload remains behind the authenticated
job artifact download route so API consumers do not accidentally move large
binary weights through ordinary status or checkpoint responses. Checkpoint
import validates the weight metadata schema, framework, format, artifact paths,
artifact sizes, SHA-256 digests, and config digest before returning it as
source metadata. When weight metadata is present, import also returns a
`studio.training.weight-restore-plan.v1` object with the owning job ID, source
status, artifact route template, loader policy, and exact artifact hashes that
clients must verify before materializing the PyTorch state dictionary. The
Training panel surfaces that restore-plan metadata after checkpoint import so
operators can inspect the source job, loader policy, route template, and
artifact hashes before any later weight materialization step. The panel can
also fetch the declared weight artifact through the authenticated job-artifact
route and verify its byte length plus SHA-256 digest against the restore plan.
After verification, operators can export a
`studio.training.weight-restore-verification.v1` manifest that records the
source job, route template, loader policy, metadata artifact hash, weight
artifact hash, byte count, and verification timestamp without embedding raw
model weights. Administrators that need an authenticated, audited
materialization can call `POST /api/studio/training/weight-restore` with the
source training job ID. The endpoint rebuilds the canonical restore plan from
the source job's stored checkpoint metadata, fetches the integrity-checked
weight and metadata artifacts, and runs the untrusted PyTorch deserialization
inside a bounded worker job (never in the request thread). The worker rechecks
artifact sizes and SHA-256 digests, loads the weights through a trusted
state-dictionary loader that restricts deserialization to tensors and primitive
containers (`torch.load(..., weights_only=True)`), and writes a path-free
`studio.training.weight-restore.v1` evidence artifact that records only the
verified digests, parameter count, and loaded-key total. The in-memory tensor
state dictionary never leaves the worker, so no raw weights reach the API
response. The Training panel surfaces that materialization evidence after the
operator triggers the restore.

## What a training request may ask for

Every request is resolved against the training contract
(`studio.training-config.v1`) before a dataset is loaded or a model is built. A
choice the Studio cannot execute is refused with HTTP 422 naming the field and
the supported set — it is never substituted.

| Field | Accepted |
|---|---|
| `dataset` | `synthetic`, `mnist` |
| `surrogate` | `fast_sigmoid`, `superspike`, `atan_surrogate`, `sigmoid_surrogate`, `straight_through`, `triangular` |
| `hidden` | A list of hidden-layer widths, each a positive integer. `[]` builds the direct input-to-output layer |
| `epochs`, `batch_size`, `timesteps` | Positive integers |
| `lr` | A positive finite number |
| `max_grad_norm` | A finite number at or above zero; `0` clips every gradient, running the loop without learning |
| `learn_beta`, `learn_threshold` | `true` or `false` |
| `seed` | An integer in `[0, 2**32)`, applied to the Python, NumPy and Torch generators |

A field nobody reads is refused too: a request carrying `hiddens` is a request
whose architecture nobody honoured, so it is rejected rather than ignored.

### Every hidden width is honoured

`hidden: [128, 64]` builds a 128-unit layer followed by a 64-unit one. It used
to build 128 twice — the request's first width repeated for as many layers as
the list was long — and the exported checkpoint recorded the request beside the
architecture that actually ran, both sealed under one digest. The two blocks
now come from the same resolved configuration and cannot disagree.

### A run is replayable from its recorded seed

`seed` is applied to the Python, NumPy and Torch generators before the loaders
and the model exist, and it is recorded in the checkpoint. Two runs with the
same seed and configuration produce identical metrics; a different seed
produces a different run. The seed lives in the request rather than in ambient
process state, so a checkpoint's seed is what actually produced it.

Those generators belong to the whole process, so a process trains one job at a
time: a run that drew from them while another trained could not be replayed.
The process-backed route gives every job its own process and never waits; a
second legacy in-process run waits until the first has finished.

### Warm start and exact resume are different runs

`POST /api/studio/training/weight-restore/attach` takes a `mode`:

| Mode | What it starts |
|---|---|
| `warm_start` (default) | A **new** run beginning from the restored weights, with a fresh optimiser and generator at epoch zero |
| `exact_resume` | A **continuation** of the source run from its recorded position: the optimiser state, the Python, NumPy and Torch generator states as they stood at the epoch boundary, and the epochs already completed |

The difference is measurable, and it is why the two are named separately: a
warm start discards Adam's moment estimates and restarts the shuffle order, so
it does not reach where the interrupted run was heading. An exact resume does.
Training two epochs, and training one then resuming for the second, produce
**identical** metrics.

A resume is refused when the saved position belongs to a different
architecture or a different configuration — asking for more epochs is not a
difference, since that is the ordinary reason to resume. A checkpoint written
by a build that recorded no position supports a warm start and says so rather
than pretending to continue anything.

The saved position travels inside the weight checkpoint under `resume_state`,
as tensors and plain integers, so the artefact still loads under
`weights_only=True`. A checkpoint arrives from a user; unpickling one is not
an option, and the generator states are stored as integers and a hex string
for exactly that reason.

`dataset_fingerprint` records what the run was trained on: a digest over the
dataset length, batch size, sample layout and the first and last samples. It
detects a different dataset, a different split boundary or a different sample
layout. It does **not** detect a change confined to the middle of a large
corpus — hashing every sample on every run would cost more than the training
step it protects.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/training/surrogates` | List available surrogate functions |
| GET | `/api/training/cell-types` | List available neuron cell types |
| POST | `/api/training/start` | Start a training job |
| POST | `/api/training/stop` | Stop a running job |
| GET | `/api/training/status/{job_id}` | Query job status |
| GET | `/api/training/checkpoint/{job_id}` | Export portable checkpoint JSON |
| POST | `/api/training/checkpoint/import` | Validate checkpoint and restore config |
| GET | `/api/training/stream/{job_id}` | SSE metric stream |
| GET | `/api/training/jobs` | List retained training jobs in creation order; pre-v7 rows have `config: null` |
| POST | `/api/studio/training/weight-restore` | Materialize and verify weights (admin) |
| POST | `/api/studio/training/weight-restore/attach` | Warm-start a job from verified weights (admin) |
| POST | `/api/studio/training/weight-restore/attach/live` | Live-attach verified weights into a running job (admin) |

### POST /api/training/start

```json
{
  "dataset": "synthetic",
  "epochs": 10,
  "batch_size": 64,
  "lr": 0.001,
  "hidden": [128],
  "timesteps": 25,
  "surrogate": "atan_surrogate",
  "learn_beta": false,
  "learn_threshold": false
}
```

Returns:

```json
{"job_id": "sj_1711504200000", "status": "running"}
```

### GET /api/training/checkpoint/{job_id}

Returns a `studio.training.checkpoint.v1` payload:

```json
{
  "schema_version": "studio.training.checkpoint.v1",
  "job_id": "sj_1711504200000",
  "status": "completed",
  "config": {"dataset": "synthetic", "epochs": 10},
  "config_sha256": "...",
  "evidence_summary": {
    "schema_version": "studio.training.evidence-summary.v1",
    "evidence_classification": "training",
    "status": "completed",
    "replay_route": "POST /api/training/start"
  },
  "weight_checkpoint": {
    "schema_version": "studio.training.weight-checkpoint.v1",
    "weights_artifact": {
      "relative_path": "training/model_state.pt",
      "size_bytes": 12345,
      "sha256": "..."
    }
  },
  "checkpoint_sha256": "..."
}
```

### POST /api/training/checkpoint/import

Accepts the checkpoint JSON and returns the validated config:

```json
{
  "imported_schema_version": "studio.training.checkpoint.v1",
  "source_job_id": "sj_1711504200000",
  "source_status": "completed",
  "config": {"dataset": "synthetic", "epochs": 10},
  "config_sha256": "...",
  "source_weight_checkpoint": {
    "schema_version": "studio.training.weight-checkpoint.v1",
    "weights_artifact": {
      "relative_path": "training/model_state.pt",
      "size_bytes": 12345,
      "sha256": "..."
    }
  },
  "weight_restore_plan": {
    "schema_version": "studio.training.weight-restore-plan.v1",
    "source_job_id": "sj_1711504200000",
    "source_status": "completed",
    "artifact_route_template": "/api/studio/jobs/{job_id}/artifacts/{artifact_path}",
    "loader_policy": "download_from_authenticated_artifact_route_and_verify_sha256",
    "restore_ready": true,
    "weights_artifact": {
      "relative_path": "training/model_state.pt",
      "size_bytes": 12345,
      "sha256": "..."
    }
  }
}
```

### POST /api/studio/training/weight-restore

Admin-only. Materializes and verifies a completed training job's weights inside
a bounded worker job and returns path-free restore evidence. Request body:

```json
{
  "source_job_id": "sj_1711504200000",
  "expected_config_sha256": "..."
}
```

`expected_config_sha256` is optional; when present it must match the source
checkpoint's configuration digest or the request is rejected with `422`.
Returns a `studio.training.weight-restore.v1` evidence object plus the worker
job ID and artifacts:

```json
{
  "schema_version": "studio.training.weight-restore.v1",
  "evidence_classification": "training",
  "status": "completed",
  "source_job_id": "sj_1711504200000",
  "source_status": "completed",
  "materialization": {
    "schema_version": "studio.training.weight-materialization.v1",
    "architecture": "64->128->10",
    "parameter_count": 9610,
    "loaded_key_count": 6,
    "config_sha256": "...",
    "weights_sha256": "...",
    "metadata_sha256": "..."
  },
  "job_id": "sj_restore_...",
  "artifacts": [
    {"relative_path": "training/weight-restore.json", "size_bytes": 256, "sha256": "..."}
  ]
}
```

Error responses: `404` when the source job is unknown, `409` when the job
published no weight checkpoint, and `422` when the restore plan or config digest
is invalid. The evidence object can be supplied to
`POST /api/studio/evidence/bundle` under `weight_restore_results` to preserve it
in an evidence bundle.

### POST /api/studio/training/weight-restore/attach

Admin-only. Warm-starts a new bounded training job seeded with the verified
weights of a completed source job. Request body:

```json
{
  "source_job_id": "sj_1711504200000",
  "config": {"dataset": "synthetic", "hidden": [128], "epochs": 5},
  "expected_config_sha256": "..."
}
```

The endpoint builds the canonical restore plan from the source checkpoint,
delivers the integrity-checked weight and metadata artifacts to the worker as
confined seed inputs, and starts a process job that materializes and verifies
the weights before loading them into the target model at the epoch-zero
checkpoint boundary (`load_state_dict(..., strict=True)`). A compatible attach
trains forward and writes a path-free `studio.training.weight-restore-attach.v1`
evidence artifact recording the verified digests, the resolved target
architecture, and the architecture fingerprint that gated compatibility. An
incompatible architecture fails the job before training begins, so partial
weights are never applied. The response is returned immediately:

```json
{
  "job_id": "sj_attach_...",
  "status": "running",
  "source_job_id": "sj_1711504200000",
  "architecture_fingerprint": "..."
}
```

The architecture fingerprint folds only the configuration fields that determine
the model state-dictionary shape (dataset, hidden widths, and the learnable
beta/threshold flags), so warm-start compatibility is independent of the
learning rate, epoch count, batch size, or timestep count. Error responses:
`404` when the source job is unknown, `409` when it published no weight
checkpoint, and `422` when the restore plan or config digest is invalid. The
attach evidence can be supplied to `POST /api/studio/evidence/bundle` under
`weight_restore_attach_results`.

### POST /api/studio/training/weight-restore/attach/live

Admin-only. Delivers the verified weights of a completed source job to a
**running process-backed** target training job, which applies them at its next
epoch boundary.
Request body:

```json
{
  "target_job_id": "sj_running_...",
  "source_job_id": "sj_1711504200000",
  "expected_config_sha256": "..."
}
```

The endpoint validates that the target is a running process worker and, when
both job configurations are known, that their architecture fingerprints match
(a mismatch is rejected with `409`). A thread-backed target or a worker that
stops during command delivery also fails closed with `409`; the API never
acknowledges a control command that cannot be consumed. It builds the canonical
restore plan from the source checkpoint and delivers the integrity-checked
weight and metadata artifacts to the running worker through the confined
control channel — a reserved control directory in the job sandbox that the
worker polls at each epoch boundary. The worker verifies and loads the weights
with a strict `load_state_dict` and writes a path-free
`studio.training.weight-restore-attach.v1` (`mode: live`) evidence artifact. An
incompatible or malformed attach is rejected with an `attach_rejected` metric
event and never interrupts the running job. The response is returned
immediately on delivery:

```json
{
  "target_job_id": "sj_running_...",
  "source_job_id": "sj_1711504200000",
  "status": "attach_requested",
  "architecture_fingerprint": "..."
}
```

Error responses: `404` when the target or source job is unknown, `409` when the
target is not running, the source published no weight checkpoint, or the
architectures are incompatible, and `422` when the restore plan or config digest
is invalid. Because the attach is applied asynchronously at the next epoch
boundary, the outcome is surfaced through the training metric stream (`attach`
or `attach_rejected` events) and the resulting evidence artifact rather than the
immediate response.

### GET /api/training/{job_id}/stream

Server-Sent Events stream. Each message is a JSON object:

```
data: {"event": "config", "data": {"job_id": "...", "device": "cuda", ...}}
data: {"event": "batch", "data": {"epoch": 0, "batch": 10, "loss": 2.31, "accuracy": 0.12}}
data: {"event": "epoch", "data": {"epoch": 0, "train_loss": 2.28, "val_loss": 2.30, "train_accuracy": 0.15, "val_accuracy": 0.13, "layer_spike_rates": {"lifs.0": 0.08}, "param_snapshot": {}}}
data: {"event": "completed", "data": {"train_loss": 1.85, "val_accuracy": 0.42}}
```

Event types: `config`, `batch`, `epoch`, `completed`, `stopped`, `error`, `heartbeat`.

## Requirements

Training requires PyTorch:

```bash
pip install sc-neurocore[research]
```

For MNIST, torchvision is also needed. Without it an MNIST run fails with
that reason before training; no other data is substituted under the MNIST
name.
