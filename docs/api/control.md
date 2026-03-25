# Neuromorphic Control

Spike-domain control theory primitives. Gains are synaptic weights, integration is membrane dynamics.

- `SpikingPID` — Population-coded PID controller. Error → rate-coded spike populations → P/I/D channels → control output. (Stagsted 2020, RSS)
- `SpikingKalmanFilter` — State estimation with spike-compatible dynamics. Predict + update cycle, Kalman gain as weight matrix.
- `SpikingLQR` — Linear Quadratic Regulator. Optimal gain computed via discrete algebraic Riccati equation. `u = -K @ x`. (SNN-LQR-EMSIF, Nature Scientific Reports 2025)

```python
from sc_neurocore.control import SpikingPID, SpikingKalmanFilter, SpikingLQR
```

See [Tutorial 79: Neuromorphic Control](../tutorials/79_control.md) for usage examples.

::: sc_neurocore.control.controllers
    options:
      show_root_heading: true
      members:
        - SpikingPID
        - SpikingKalmanFilter
        - SpikingLQR
