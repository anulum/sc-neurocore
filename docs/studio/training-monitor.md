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
Events). Each epoch emits:

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
- **LapicqueCell** — classical Lapicque neuron
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
idle → starting → running → completed | stopped | failed
```

Multiple training jobs can run concurrently. Each job has a unique ID
used for status queries and SSE stream subscription.

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
| GET | `/api/training/jobs` | List all jobs |
| POST | `/api/studio/training/weight-restore` | Materialize and verify weights (admin) |

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

For MNIST, torchvision is also needed. If unavailable, synthetic data
is used as fallback.
