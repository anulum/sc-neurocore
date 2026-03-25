# Tutorial 82: Spiking Neural ODEs

Continuous-depth SNN layer: adaptive ODE solver with event-driven spike
detection. Takes large steps when membrane is far from threshold, bisects
on crossings for sub-timestep precision. No other library has this as a
reusable layer.

## How It Works

1. Large steps when membrane is far from threshold
2. Shrink step size near threshold crossings
3. Bisection to find exact spike times
4. Reset and continue after spike emission

## SpikingODELayer

```python
import numpy as np
from sc_neurocore.spike_ode import SpikingODELayer, ODELIFDynamics

dynamics = ODELIFDynamics(
    tau_mem=20.0, v_rest=0.0, v_threshold=1.0, v_reset=0.0,
)

layer = SpikingODELayer(
    n_inputs=32, n_neurons=16,
    dynamics=dynamics, dt_init=0.1, dt_min=0.001,
)

inputs = np.random.randn(100, 32) * 0.5
spike_counts = layer.forward(inputs, interval=1.0)
# shape: (100, 16)
print(f"Total spikes: {int(spike_counts.sum())}")
```

## Single-Step (Online)

```python
layer.reset()
x = np.random.randn(32) * 0.5
counts = layer.step(x, interval=1.0)
print(f"Voltage: {layer.voltage}")
```

Adaptive stepping gives ~5x speedup over fixed small-step integration.

## API Reference

::: sc_neurocore.spike_ode.ode_layer
    options:
      show_root_heading: true
