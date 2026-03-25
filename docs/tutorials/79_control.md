# Tutorial 79: Neuromorphic Control

Spike-domain control theory: PID, Kalman filter, LQR. All controllers use
population-coded spike representations. No other SNN library provides
control-theory primitives.

## Spiking PID Controller

```python
import numpy as np
from sc_neurocore.control import SpikingPID

pid = SpikingPID(Kp=1.0, Ki=0.1, Kd=0.01, n_neurons=10, dt=0.01)

setpoint, measurement = 1.0, 0.0
for step in range(200):
    error = setpoint - measurement
    control = pid.step(error)
    measurement += control * 0.01

# Spike-domain output: population-coded P/I/D channels
rng = np.random.RandomState(42)
spike_output = pid.step_spike(error=0.5, rng=rng)
# shape: (30,) = [P(10), I(10), D(10)]
```

## Spiking Kalman Filter

```python
from sc_neurocore.control import SpikingKalmanFilter

kf = SpikingKalmanFilter(
    n_states=4, n_measurements=2,
    A=np.array([[1,0,0.1,0],[0,1,0,0.1],[0,0,1,0],[0,0,0,1]]),
    H=np.array([[1,0,0,0],[0,1,0,0]]),
)

for t in range(50):
    z = np.array([t*0.1 + np.random.randn()*0.1, t*0.05 + np.random.randn()*0.1])
    state = kf.step(z)
```

## Spiking LQR

```python
from sc_neurocore.control import SpikingLQR

A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.0], [0.1]])
lqr = SpikingLQR(A=A, B=B)

x = np.array([1.0, 0.0])
for t in range(100):
    u = lqr.control(x)
    x = A @ x + B @ u
# x converges to origin
```

## API Reference

::: sc_neurocore.control.controllers
    options:
      show_root_heading: true
