# Tutorial 79: Neuromorphic Control

Spike-domain control theory: PID controllers, Kalman filters, and LQR
regulators implemented with population-coded spike representations.
Every control signal is a spike train — enabling deployment on
neuromorphic hardware with microsecond latency and microwatt power.

No other SNN library provides control-theory primitives in spike domain.

## Why Spike-Based Control

Traditional control: sensor → ADC → digital controller → DAC → actuator.
Spike control: sensor → spike encoder → spike controller → actuator.

| Property | Digital PID | Spike PID |
|----------|-----------|-----------|
| Latency | ~1 ms (ADC + compute + DAC) | ~10 µs (spike propagation) |
| Power | ~10 mW (microcontroller) | ~10 µW (neuromorphic) |
| Update rate | Fixed clock (1-10 kHz) | Event-driven (asynchronous) |
| Resolution | Fixed (12-16 bit ADC) | Adaptive (population coding) |

For robotics, prosthetics, and autonomous drones, spike-based control
offers 100× lower latency and 1000× lower power.

## Spiking PID Controller

Population-coded PID with separate neural populations for P, I, and D
channels:

```python
import numpy as np
from sc_neurocore.control import SpikingPID

pid = SpikingPID(
    Kp=1.0,         # proportional gain
    Ki=0.1,         # integral gain
    Kd=0.01,        # derivative gain
    n_neurons=10,   # neurons per channel (30 total: P+I+D)
    dt=0.01,        # timestep (10 ms)
)

# Setpoint tracking simulation
setpoint = 1.0
measurement = 0.0
trajectory = []

for step in range(200):
    error = setpoint - measurement
    control = pid.step(error)
    measurement += control * 0.01  # simple plant: integrator
    trajectory.append(measurement)

print(f"Final value: {trajectory[-1]:.4f}")
print(f"Settling time: {next(i for i, v in enumerate(trajectory) if abs(v - 1.0) < 0.05) * 10} ms")

# Spike-domain output: population-coded P/I/D channels
rng = np.random.RandomState(42)
spike_output = pid.step_spike(error=0.5, rng=rng)
print(f"Spike output shape: {spike_output.shape}")  # (30,) = [P(10), I(10), D(10)]
print(f"P channel rate: {spike_output[:10].mean():.2f}")
print(f"I channel rate: {spike_output[10:20].mean():.2f}")
print(f"D channel rate: {spike_output[20:30].mean():.2f}")
```

### Population Coding

Each control channel (P, I, D) uses a population of neurons with
different preferred values. The population spike rate encodes the
control signal magnitude:

```
Control value 0.5 → population fires at ~50% of max rate
Control value 1.0 → population fires at ~100% of max rate
Control value 0.0 → population is silent
```

This provides graceful degradation (losing a neuron reduces resolution,
doesn't crash the controller) and natural noise filtering.

## Spiking Kalman Filter

State estimation from noisy measurements using spike-based
predict-update cycles:

```python
from sc_neurocore.control import SpikingKalmanFilter

# 2D tracking: state = [x, y, vx, vy], measurement = [x, y]
A = np.array([[1, 0, 0.1, 0],    # x += vx * dt
              [0, 1, 0, 0.1],    # y += vy * dt
              [0, 0, 1, 0],      # vx constant
              [0, 0, 0, 1]])     # vy constant

H = np.array([[1, 0, 0, 0],      # observe x
              [0, 1, 0, 0]])     # observe y

kf = SpikingKalmanFilter(
    n_states=4,
    n_measurements=2,
    A=A, H=H,
)

# Track a moving target with noisy measurements
true_x, true_y = 0.0, 0.0
vx, vy = 0.1, 0.05
estimates = []

for t in range(100):
    true_x += vx
    true_y += vy
    # Noisy measurement
    z = np.array([true_x + np.random.randn() * 0.1,
                  true_y + np.random.randn() * 0.1])
    state = kf.step(z)
    estimates.append(state[:2].copy())

error = np.sqrt((estimates[-1][0] - true_x)**2 + (estimates[-1][1] - true_y)**2)
print(f"Final tracking error: {error:.4f}")
```

## Spiking LQR (Linear Quadratic Regulator)

Optimal control for linear systems with quadratic cost:

```python
from sc_neurocore.control import SpikingLQR

# Inverted pendulum: state = [angle, angular_velocity]
A = np.array([[1.0, 0.1],   # angle += omega * dt
              [0.0, 1.0]])  # omega += torque * dt
B = np.array([[0.0],
              [0.1]])       # torque input

lqr = SpikingLQR(A=A, B=B)

# Stabilise from initial displacement
x = np.array([0.5, 0.0])  # 0.5 rad initial angle
trajectory = [x.copy()]

for t in range(200):
    u = lqr.control(x)
    x = A @ x + B @ u
    trajectory.append(x.copy())

print(f"Initial angle: {trajectory[0][0]:.3f} rad")
print(f"Final angle:   {trajectory[-1][0]:.6f} rad")
print(f"Converged: {abs(trajectory[-1][0]) < 0.001}")
```

## FPGA Deployment

All controllers map to small FPGA designs:

| Controller | LUTs (est.) | Latency | Power |
|-----------|-------------|---------|-------|
| Spiking PID (30 neurons) | ~200 | 1 clock cycle | ~50 µW |
| Spiking Kalman (4 states) | ~500 | 4 clock cycles | ~200 µW |
| Spiking LQR (2 states) | ~300 | 2 clock cycles | ~100 µW |

```python
# Export controller for FPGA
pid.export_fpga("pid_ice40.v", target="ice40")
# Generates synthesisable SystemVerilog with fixed-point coefficients
```

## Applications

| Application | Controller | Why Spike |
|-------------|-----------|-----------|
| Robotic arm | PID | Microsecond response, low power |
| Drone stabilisation | LQR | Optimal control at microwatt power |
| BCI cursor | Kalman | State estimation from neural spikes |
| Prosthetic hand | PID + Kalman | Sensory feedback + motor control |
| Autonomous vehicle | LQR + Kalman | Sensor fusion + path tracking |

## References

- DeWolf et al. (2016). "A spiking neural model of adaptive arm
  control." Proc. R. Soc. B 283:20162134.
- Eliasmith & Anderson (2003). "Neural Engineering: Computation,
  Representation, and Dynamics in Neurobiological Systems." MIT Press.
- Stagsted et al. (2020). "Towards neuromorphic control: A spiking
  neural network based PID controller for UAV." RSS 2020.
