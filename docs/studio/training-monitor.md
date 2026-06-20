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

Training runs in a background thread. The lifecycle is:

```
idle → starting → running → completed | stopped | failed
```

Multiple training jobs can run concurrently. Each job has a unique ID
used for status queries and SSE stream subscription.

### Portable Checkpoints

Training checkpoints use the `studio.training.checkpoint.v1` schema. They are
portable JSON manifests for Studio configuration, terminal status, metrics, and
evidence metadata. They do not expose local filesystem paths and do not include
raw model-weight tensors. Import validates both the config digest and full
checkpoint digest before returning the restored training config to the UI.

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
  "config_sha256": "..."
}
```

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
