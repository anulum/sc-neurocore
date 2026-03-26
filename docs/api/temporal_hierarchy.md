# Multi-Timescale SNN — Heterogeneous Synapses + Multi-Clock

Per-synapse learnable time constants and multi-clock layer scheduling. Biological brains have timescales spanning 5 orders of magnitude (1ms–10s); this module enables the same in simulation.

## HetSynLayer — Heterogeneous Synaptic Time Constants

Each synapse has its own tau, initialized log-normally (matching Allen Institute cortical data). Different synapses integrate over different temporal windows, enabling a single layer to capture both fast transients and slow trends.

`trace[i,j] = exp(-dt/tau[i,j]) * trace[i,j] + input_spike[j]`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_inputs` | (required) | Input dimension |
| `n_neurons` | (required) | Output dimension |
| `tau_mean` | 5.0 | Mean synaptic time constant (ms) |
| `tau_std` | 1.0 | Std of log(tau) for log-normal init |
| `threshold` | 1.0 | LIF spike threshold |

The `tau_stats` property returns `{mean, std, min, max, median}` of the tau distribution.

## MultiClockSNN — Multi-Clock Scheduling

Different layers run at different temporal resolutions. Fast sensory layers tick every step, slow cognitive layers tick every N steps. Between ticks, layers hold their last output (clock-domain crossing buffer).

| Parameter | Meaning |
|-----------|---------|
| `layers` | List of HetSynLayer |
| `layer_names` | Names for each layer |
| `clock_intervals` | Steps between updates per layer (default all 1) |

Methods: `step(x, dt)`, `run(inputs, dt)`, `reset()`.

## Usage

```python
from sc_neurocore.temporal_hierarchy.multi_clock import HetSynLayer, MultiClockSNN
import numpy as np

# Fast sensory layer (tick every step)
sensory = HetSynLayer(n_inputs=32, n_neurons=64, tau_mean=2.0)

# Slow cognitive layer (tick every 10 steps)
cognitive = HetSynLayer(n_inputs=64, n_neurons=16, tau_mean=50.0)

# Multi-clock network
net = MultiClockSNN(
    layers=[sensory, cognitive],
    layer_names=["sensory", "cognitive"],
    clock_intervals=[1, 10],
)

# Run 100 timesteps
inputs = np.random.randn(100, 32)
outputs = net.run(inputs, dt=1.0)  # (100, 16)

# Check tau distribution
print(sensory.tau_stats)
```

**Reference:** HetSyn (NeurIPS 2025).

::: sc_neurocore.temporal_hierarchy
    options:
      show_root_heading: true
