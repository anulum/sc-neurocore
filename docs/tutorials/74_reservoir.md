# Tutorial 74: Auto-Critical Reservoir Computing

Liquid State Machines (LSMs) and Echo State Networks use a fixed
random recurrent network (reservoir) with only the readout layer
trained. SC-NeuroCore's auto-critical reservoir self-tunes to the
edge of chaos — the regime where computational capacity is maximised
— using mean-field theory. Zero hyperparameter tuning needed.

## Why Reservoir Computing

| Property | Backprop SNN | Reservoir SNN |
|----------|-------------|---------------|
| Training | All weights (slow) | Readout only (fast) |
| Memory cost | O(W^2) gradients | O(W_readout) only |
| Temporal processing | Learned | Emergent from dynamics |
| Hardware deployment | Weight updates everywhere | Fixed reservoir, simple readout |
| Biological plausibility | Low | High (cortical microcircuits) |

Reservoirs are ideal for temporal pattern recognition where the input
structure contains the information and you just need a trained decoder.

## AutoCriticalReservoir

```python
import numpy as np
from sc_neurocore.reservoir import AutoCriticalReservoir

# Create reservoir: 64 inputs, 1000 recurrent neurons, 10 outputs
reservoir = AutoCriticalReservoir(
    n_inputs=64,
    n_neurons=1000,
    n_outputs=10,
)

# Generate data
rng = np.random.default_rng(42)
train_x = rng.standard_normal((500, 64)).astype(np.float32)
train_y = np.eye(10, dtype=np.float32)[rng.integers(0, 10, 500)]
test_x = rng.standard_normal((100, 64)).astype(np.float32)

# Train readout and predict (reservoir weights are fixed)
predictions = reservoir.train_and_predict(train_x, train_y, test_x)
print(f"Predictions shape: {predictions.shape}")  # (100, 10)

# Check criticality metrics
metrics = reservoir.metrics(test_x)
print(metrics.summary())
# Spectral radius: 0.98 (near 1.0 = edge of chaos)
# Lyapunov exponent: -0.02 (slightly negative = stable chaos)
# Memory capacity: 47.3 / 64 (74% of theoretical maximum)
# Separation ratio: 0.89 (good input discrimination)
```

## How Auto-Criticality Works

Standard reservoirs require manual tuning of:
- **Spectral radius** — eigenvalue magnitude of recurrent weight matrix
- **Input scaling** — how strongly inputs drive the reservoir
- **Leak rate** — membrane decay time constant
- **Sparsity** — fraction of non-zero recurrent connections

The auto-critical reservoir uses mean-field theory to compute the
optimal spectral radius analytically:

```
sigma_optimal = 1 / sqrt(p * N * <f'(x)^2>)
```

Where p is connection probability, N is neuron count, and `<f'(x)^2>`
is the mean squared derivative of the activation function, estimated
from a short probe run.

This keeps the reservoir at the **edge of chaos** — the phase
transition between ordered (too stable, no computation) and chaotic
(too unstable, no memory) dynamics. Computational capacity is
maximised at this critical point.

## Temporal Classification

For time-series classification (speech, gestures, EEG):

```python
# Temporal data: 500 samples, 100 timesteps, 64 features
temporal_x = rng.standard_normal((500, 100, 64)).astype(np.float32)
temporal_y = np.eye(5, dtype=np.float32)[rng.integers(0, 5, 500)]

# The reservoir processes each timestep sequentially
# Final reservoir state encodes the temporal pattern
reservoir = AutoCriticalReservoir(n_inputs=64, n_neurons=500, n_outputs=5)
predictions = reservoir.train_and_predict_temporal(temporal_x, temporal_y, test_temporal_x)
```

## FPGA Deployment

Reservoirs are ideal for FPGA because:
1. **Fixed weights** — no weight updates during inference → no write ports
2. **Sparse connectivity** — CSR storage uses minimal BRAMs
3. **Simple readout** — one dense layer, easily pipelined

```python
# Export for FPGA
reservoir.export_weights("reservoir_ice40.npz")

# In the Studio:
# Import the reservoir as a network on the Canvas
# The recurrent population has fixed weights (no plasticity)
# Only the readout projection is configurable
# Pipeline → ice40 for synthesis
```

## Comparison

| Feature | SC-NeuroCore | Reservoirpy | NEST (LSM) | Brian2 |
|---------|:-----------:|:-----------:|:----------:|:------:|
| Auto-criticality | Yes | No | No | No |
| Spiking dynamics | LIF | Rate | LIF | LIF |
| FPGA deployment | Yes | No | No | No |
| Memory capacity metric | Yes | Yes | No | No |
| Zero tuning needed | Yes | No | No | No |

## References

- Maass et al. (2002). "Real-Time Computing Without Stable States: A
  New Framework for Neural Computation Based on Perturbations." Neural
  Computation 14(11):2531-2560.
- Bertschinger & Natschläger (2004). "Real-Time Computation at the Edge
  of Chaos in Recurrent Neural Networks." Neural Computation 16(7).
- Lukoševičius & Jaeger (2009). "Reservoir computing approaches to
  recurrent neural network training." Computer Science Review 3(3).

## Interactive Notebook

Run the hands-on notebook: [`notebooks/17_reservoir_computing.ipynb`](https://github.com/anulum/sc-neurocore/blob/main/notebooks/17_reservoir_computing.ipynb)
