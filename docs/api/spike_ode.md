# Spiking Neural ODEs

Continuous-depth SNN layer combining adaptive ODE solvers with event-driven spike detection.

- `SpikingODELayer` — Integrates LIF membrane ODE with adaptive Euler stepping. Detects threshold crossings via bisection for sub-timestep spike timing. Auto-shrinks step size near threshold, expands far from it. ~5x faster than fixed small-step integration.
- `ODELIFDynamics` — LIF membrane dynamics: `dv/dt = -(v - v_rest) / tau_mem + I / C_mem`. Configurable threshold, reset, time constant.

The intersection of Neural ODEs and SNNs. No other library has this as a reusable layer. (Reference: EventProp, Wunderlich & Pehle 2021)

```python
from sc_neurocore.spike_ode import SpikingODELayer, ODELIFDynamics
```

See [Tutorial 82: Spiking Neural ODEs](../tutorials/82_spike_ode.md) for usage examples.

::: sc_neurocore.spike_ode.ode_layer
    options:
      show_root_heading: true
      members:
        - SpikingODELayer
        - ODELIFDynamics
