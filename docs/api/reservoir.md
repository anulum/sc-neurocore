# Reservoir Computing — Auto-Critical LSM

Liquid State Machine with mean-field auto-criticality tuning. Zero hyperparameter tuning: the critical weight `W_c = θ / (2βNp)` is computed analytically from threshold (θ), leak (β), neuron count (N), and connectivity (p).

## Theory

At the critical regime, exactly half the neurons fire in each refractory period, maximizing the reservoir's computational capacity (kernel quality). The reservoir projects inputs into a high-dimensional nonlinear state space; only the readout layer (ridge regression) is trained.

The critical weight formula derives from mean-field analysis: the expected number of post-synaptic spikes per pre-synaptic spike equals 1.0 at criticality, producing a branching ratio of exactly 1.

## Components

- **`AutoCriticalReservoir`** — Full LSM pipeline with auto-criticality tuning.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_inputs` | (required) | Input dimension |
| `n_neurons` | 1000 | Reservoir size |
| `n_outputs` | 10 | Readout dimension |
| `threshold` | 1.0 | LIF spike threshold |
| `leak` | 0.1 | Membrane leak factor (0-1) |
| `connectivity` | 0.1 | Synapse existence probability |
| `seed` | 42 | RNG seed |

**Key methods:**

- `step(x)` — Process one timestep, return spike vector
- `run(inputs)` — Run full input sequence (T, n_inputs) → state matrix (T, n_neurons)
- `fit_readout(states, targets, ridge)` — Train readout via ridge regression
- `predict(states)` — Predict from reservoir states
- `train_and_predict(train_in, train_tgt, test_in)` — Full pipeline
- `metrics(inputs)` → `ReservoirMetrics` — firing fraction, criticality error, kernel quality, spectral radius

- **`ReservoirMetrics`** — Quality metrics dataclass with `summary()` method.

## Usage

```python
from sc_neurocore.reservoir import AutoCriticalReservoir

# Zero-config reservoir
res = AutoCriticalReservoir(n_inputs=2, n_neurons=500)
print(f"Spectral radius: {res.spectral_radius:.3f}")
print(f"Critical weight: {res.w_critical:.6f}")

# Run and train
import numpy as np
train_in = np.random.randn(200, 2)
train_tgt = np.sin(np.arange(200)).reshape(-1, 1)
test_in = np.random.randn(50, 2)

preds = res.train_and_predict(train_in, train_tgt, test_in)

# Quality assessment
metrics = res.metrics(train_in)
print(metrics.summary())
```

**Reference:** Scientific Reports 2025 — mean-field analytical framework for configuring spiking reservoirs at the critical regime.

See [Tutorial 74: Reservoir Computing](../tutorials/74_reservoir.md).

::: sc_neurocore.reservoir
    options:
      show_root_heading: true
