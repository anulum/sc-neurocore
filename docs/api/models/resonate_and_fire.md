# ResonateAndFireNeuron

**Module:** `sc_neurocore.neurons.models.resonate_and_fire`
**Rust:** `sc_neurocore_engine::neurons::simple_spiking::ResonateAndFireNeuron`
**Reference:** Izhikevich, E. M. (2001)
**Publication:** *Resonate-and-fire neurons.* Neural Networks, 14(6-7), 883–894.
**Family:** Oscillatory (complex-valued subthreshold dynamics)
**State variables:** `x`, `y` (real and imaginary parts of complex state z = x + iy)

---

## Equations

### Complex form

$$\frac{dz}{dt} = (b + i\omega)\,z + I$$

where $z = x + iy$, $b$ is the damping/growth rate, $\omega$ is the natural
oscillation frequency, and $I$ is real-valued input current.

### Decomposed into real ODEs (as implemented)

$$\frac{dx}{dt} = bx - \omega y + I$$
$$\frac{dy}{dt} = \omega x + by$$

### Spike condition

$$|z| = \sqrt{x^2 + y^2} \geq \theta$$

### Reset

On spike: $x \leftarrow 0,\; y \leftarrow 0$ (reset to origin).

### Exact implementation (as coded)

```python
def step(self, current: float) -> int:
    x_ss = -self.b * current / (self.b**2 + self.omega**2)
    y_ss = self.omega * current / (self.b**2 + self.omega**2)
    decay = math.exp(self.b * self.dt)
    angle = self.omega * self.dt
    dx = self.x - x_ss
    dy = self.y - y_ss
    next_x = x_ss + decay * (dx * math.cos(angle) - dy * math.sin(angle))
    next_y = y_ss + decay * (dx * math.sin(angle) + dy * math.cos(angle))
    radius = math.hypot(next_x, next_y)
    if not all(math.isfinite(value) for value in (next_x, next_y, radius)):
        raise ValueError("exact resonator update must be finite")
    if radius >= self.threshold:
        self.x = 0.0
        self.y = 0.0
        return 1
    self.x = next_x
    self.y = next_y
    return 0
```

The step is the closed-form constant-input solution:

$$z(t+\Delta t) = z_{ss} + e^{(b+i\omega)\Delta t}(z(t)-z_{ss})$$

with $z_{ss}=-I/(b+i\omega)$. No sub-stepping is required.

Runtime contract: all maintained Python, Julia, Go, Mojo, and Rust
safety surfaces reject non-finite current/state and non-finite exact-flow
updates before mutating oscillator state. Rejected native scalar paths
return an explicit error/sentinel rather than silently converting
numerical corruption into a no-spike event.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `x` | 0.0 | Real part of complex state |
| `y` | 0.0 | Imaginary part of complex state |
| `b` | −0.1 | Damping rate (b<0: damped, b>0: unstable) |
| `omega` | 1.0 | Natural oscillation frequency (rad/time) |
| `threshold` | 1.0 | Spike threshold on |z| |
| `dt` | 0.05 | Integration time step |

---

## Analytical Properties

### Steady-state (constant input, subthreshold)

Setting $dx/dt = 0$, $dy/dt = 0$:

$$x_{ss} = \frac{-bI}{b^2 + \omega^2}, \quad y_{ss} = \frac{I\omega}{b^2 + \omega^2}$$

$$r_{ss} = \sqrt{x_{ss}^2 + y_{ss}^2} = \frac{I}{\sqrt{b^2 + \omega^2}}$$

### Critical current

Spike occurs when $r_{ss} \geq \theta$:

$$I_{crit} = \theta \sqrt{b^2 + \omega^2}$$

With defaults ($b = -0.1$, $\omega = 1.0$, $\theta = 1.0$):

$$I_{crit} = \sqrt{0.01 + 1.0} = \sqrt{1.01} \approx 1.005$$

Verified: I=0.5 produces 0 spikes (r_ss ≈ 0.498 < 1.0). I=1.0 fires
(r_ss ≈ 0.995 but transient overshoot crosses threshold).

### Transient overshoot

The spiral approach to equilibrium can transiently exceed $r_{ss}$.
Measured: I = 0.9 × I_crit still produces spikes (2000 in 50k steps)
due to transient overshoot. A 50% margin below I_crit is needed to
guarantee zero spikes.

### Eigenvalues

The system matrix $A = \begin{pmatrix} b & -\omega \\ \omega & b \end{pmatrix}$
has eigenvalues $\lambda = b \pm i\omega$.

- $b < 0$: stable spiral (damped oscillation)
- $b = 0$: centre (undamped, marginally stable)
- $b > 0$: unstable spiral (amplitude grows exponentially)

### Oscillation period

$$T = \frac{2\pi}{\omega}$$

With $\omega = 1.0$, $T \approx 6.28$ time units, or $T/dt \approx 126$ steps.

### Biological significance

The resonate-and-fire model captures a fundamental property of many
cortical and thalamic neurons: **subthreshold resonance**. These neurons
preferentially respond to inputs at a specific frequency — they act as
bandpass filters rather than low-pass integrators. This is mediated by
voltage-dependent currents (I_h, I_M, I_NaP) that create the
subthreshold oscillation captured abstractly by the complex eigenvalue
$b \pm i\omega$. Neurons with prominent subthreshold resonance include
stellate cells in entorhinal cortex, thalamic relay neurons, and
inferior olive neurons.

---

## Behaviour

### Damped subthreshold oscillation (b < 0)

With b=−0.1, the system is a damped oscillator. Under constant subthreshold
input, x and y spiral inward toward the equilibrium point. The oscillation
is clearly visible in the x trace: > 20 zero-crossings of the mean in
4000 post-transient steps at I=0.5.

### Threshold on radius

Spike detection uses the Euclidean norm $|z| = \sqrt{x^2 + y^2}$, not a
simple voltage threshold. This means the neuron can spike via oscillation
build-up in either x or y — it responds to both amplitude and phase of
the input relative to its internal oscillation.

### Unstable regime (b > 0)

When b > 0, any perturbation from origin grows exponentially. Even with
I=0, a tiny initial displacement (x=0.01) eventually reaches threshold
and triggers spikes. Verified: b=0.1 with x_0=0.01 produces spikes with
zero input.

### omega controls oscillation frequency

Higher omega → faster subthreshold oscillation. Measured via zero-crossings
of x(t) around its mean:
- omega=0.5: fewer crossings
- omega=2.0: more crossings

This confirms the model's resonant property: it preferentially responds
to inputs oscillating near its natural frequency omega.

---

## Measured Dynamics (from test probing)

### Constant current sweep (default parameters)

| Current | Spikes (50k) | Mean ISI | r at end | Regime |
|---------|-------------|----------|----------|--------|
| 0.0 | 0 | — | 0.0000 | Origin rest |
| 0.5 | 0 | — | 0.4975 | Subthreshold spiral |
| 1.0 | 2,173 | 23 | 0.9516 | Spiking (just above I_crit) |
| 2.0 | 4,545 | 11 | 0.4925 | Regular spiking |
| 5.0 | 10,000 | 5 | 0.0000 | Fast spiking |
| 10.0 | 16,666 | 3 | 0.9946 | Rapid spiking |
| 20.0 | 25,000 | 2 | 0.0000 | Alternating-step spiking |
| 25.0 | 50,000 | 1 | 0.0000 | Every-step spiking |

f–I is monotonic. At I=25, the exact one-step radius exceeds threshold after each reset, so spike rate = 1/dt.

### Damping parameter sweep (I=1.5)

| b | Spikes (50k) | Description |
|---|-------------|-------------|
| −0.05 | many | Weak damping, lower effective I_crit |
| −0.5 | fewer | Heavier damping, higher effective I_crit |
| +0.1 | spikes even at I=0 | Unstable spiral |

---

## Comparison with Other Models

| Property | LIF | QIF | Resonate-and-Fire |
|----------|-----|-----|-------------------|
| State variables | 1 (V) | 1 (V) | 2 (x, y) |
| Subthreshold dynamics | Exponential decay | Stable/unstable FP | Damped oscillation |
| Excitability | Type-I (linear onset) | Type-I (sqrt onset) | Type-II (resonance) |
| Spike detection | V ≥ θ | V ≥ V_peak | \|z\| ≥ θ |
| Input selectivity | None (integrator) | None (integrator) | Frequency-selective |
| Reset | V → V_reset | V → V_reset | (x,y) → (0,0) |

The key distinction: R&F is a **resonator**, not an integrator. It responds
preferentially to inputs near its natural frequency ω, making it suitable
for modelling neurons in sensory systems that exhibit band-pass filtering.

---

## Numerical Considerations

- **dt stability:** Tested at dt = 0.02, 0.05, 0.1. All produce finite
  states after 50k steps at I=2.0.
- **Exact linear flow:** The maintained implementation evaluates the
  matrix-exponential solution for the damped oscillator instead of relying on
  explicit Euler stability. The homogeneous radius decays exactly by
  `exp(b*dt)` when `current = 0`, so large timesteps preserve the analytical
  damped-rotation envelope instead of introducing numerical growth.
- **Candidate validation:** Runtime code rejects non-finite exact-flow
  coefficients, equilibria, rotations, candidate coordinates, and radius
  before mutating state.
- **Radius computation:** Uses `math.hypot(x, y)` each step to avoid
  avoidable overflow in the Euclidean norm.

---

## Validation Contract

- `x`, `y`, `b`, `omega`, `threshold`, `dt`, and runtime `current` must
  be finite.
- `omega`, `threshold`, and `dt` must be strictly positive. Zero natural
  frequency is rejected because this model is a damped resonator, not a
  plain integrator.
- Each step computes `dx`, `dy`, candidate coordinates, and candidate
  radius before mutation. Python raises `ValueError` for non-finite
  candidates; Rust, Go, Julia, and Mojo fail closed without mutating state
  or reporting a spike.
- `reset()` clears only dynamic oscillator coordinates and preserves
  physical parameters.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/resonate_and_fire.py` — 57 lines.
- **Two real state variables:** x and y, representing Re(z) and Im(z).
- **Polyglot surfaces:** Python, Rust, Go, Julia, and Mojo implement the
  same finite-state, positive-frequency, positive-threshold,
  positive-timestep, spike-reset, and parameter-preserving reset contract.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary, 2-var evolution, finite 50k, reset |
| Steady-state | 3 | r_ss = I/sqrt(b²+ω²) at I=0.5, I=0.3; damping decay at b=−0.5 |
| Threshold | 4 | I_crit analytical (50% below → 0 spikes, 20% above → >10 spikes), radius check, reset to origin, custom threshold |
| f–I curve | 4 | monotonic (4-point), excess current scaling, zero input silent, I=20 fires every step |
| Oscillation | 2 | subthreshold x oscillation (>20 zero-crossings), omega frequency scaling |
| Parameters | 5 | b>0 unstable (spikes at I=0), b more negative → fewer spikes, dt stability (3 values) |
| Determinism | 1 | bit-exact (300 steps) |
| ISI | 1 | constant ISI (CV<0.05) |
| Network | 2 | Population(n=10), Network spikes |
| Analysis | 2 | spike_count ≥ 100, consistency |
| Validation | 27 | non-finite parameters, positive threshold/dt, positive omega, finite current, finite candidate update |
| **Total** | **56** | |

---

## Findings

1. **Analytical r_ss confirmed:** At I=0.5, measured r = 0.4975, predicted
   r_ss = 0.498. At I=0.3, convergence also matches within 0.01.
2. **Transient overshoot crosses threshold:** At I = 0.9 × I_crit, the
   transient spiral approach overshoots r_ss and triggers spikes. The
   analytical I_crit is a steady-state prediction, not a transient one.
   A 50% safety margin below I_crit is needed for guaranteed silence.
3. **b > 0 unstable confirmed:** With b=0.1 and x_0=0.01, the expanding
   spiral reaches threshold and fires even with zero input.
4. **Damping controls effective threshold:** More negative b raises the
   effective I_crit (heavier damping attenuates the state more).
   Measured at I=1.5: b=−0.05 fires more than b=−0.5.
5. **Omega sets oscillation frequency:** Higher omega produces more
   zero-crossings in the subthreshold x trace, confirming the
   resonance property.
6. **ISI regular at steady state:** CV(ISI) < 0.05 at I=2.0 after
   skipping the first 10 spikes (transient).


---

## Measured Performance (2026-06-01)

| Metric | Value | Notes |
|--------|-------|-------|
| Python exact-flow step | 2,742.56363 ns/step median | `200,000` steps × 5 repeats |
| Benchmark command | `PYTHONPATH=src .venv/bin/python benchmarks/bench_model_resonate_and_fire.py` | local workstation |
| Spikes per repeat | 18,181 | current = 2.0 |
| Ending state | `x=0.8509771070439969`, `y=0.1932519645494586` | deterministic across repeats |
| Closed-form transcendental calls | 4 per step | `exp`, `cos`, `sin`, `hypot` |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`ResonateAndFireNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
2000 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(ResonateAndFireNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust safety counterpart
The maintained Rust safety surface now implements the same scalar
validation, spike reset, and parameter-preserving reset contracts. This
slice did not rerun a standalone benchmark for this model.

---

## Findings (measured 2026-04-04)

1. Throughput: ~105K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust safety counterpart: scalar contract aligned
4. Numerical stability confirmed over 20K steps

---

## Pipeline Position

```
sc_neurocore Pipeline
├── Python layer
│   └── sc_neurocore.neurons.models.resonate_and_fire.ResonateAndFireNeuron
│       ├── step(current) → int {0, 1}
│       ├── reset() → None  (x=0, y=0)
│       ├── Population(ResonateAndFireNeuron, n=N)
│       ├── Network(pop, drive, monitor)
│       └── Analysis: spike_count(), firing_rate(), isi()
│
├── Rust engine
│   └── sc_neurocore_engine::neurons::simple_spiking::ResonateAndFireNeuron
│       ├── new() → Self
│       ├── step(&mut self, current: f64) → i32
│       └── reset(&mut self)
│
├── PyO3 binding
│   └── sc_neurocore_engine.ResonateAndFireNeuron (Python class)
│       ├── __init__()
│       ├── step(current) → int
│       ├── reset()
│       └── get_state() → dict {x, y}
│
└── Network runner
    └── NeuronVariant::ResonateAndFire(ResonateAndFireNeuron)
        ├── Wired in network_runner.rs
        ├── Voltage access via n.x
        └── Factory: "ResonateAndFire" | "ResonateAndFireNeuron" → new()
```

---

## Technical Reference

### Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `x` | 0.0 | — | Real part of state (initial) |
| `y` | 0.0 | — | Imaginary part of state (initial) |
| `b` | -0.1 | ms⁻¹ | Decay rate (damping, must be < 0 for stability) |
| `omega` | 1.0 | rad/ms | Natural oscillation frequency |
| `threshold` | 1.0 | — | Spike threshold on amplitude |z| |
| `dt` | 0.05 | ms | Integration timestep |

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `step` | `(current: f64) → i32` | 0 or 1 | Advance one timestep; reset on |z| ≥ threshold |
| `reset` | `() → ()` | — | Reset x=0, y=0 |
| `new` | `() → Self` | — | Rust constructor with defaults |
| `get_state` | `() → dict` | x, y | PyO3 only: state inspection |

### Python/Rust Implementation Comparison

| Aspect | Python | Rust |
|--------|--------|------|
| Source | `resonate_and_fire.py` (44 lines) | `simple_spiking.rs:186-225` |
| Integration | Simultaneous Euler | Simultaneous Euler (fixed 775e3bd) |
| sqrt per step | 1 (numpy) | 1 (f64::sqrt) |
| Amplitude check | numpy.sqrt(x²+y²) | (x*x + y*y).sqrt() |
| **Parity** | **EXACT** (after simultaneous Euler fix) | |

### NeuronVariant Wiring

```rust
// network_runner.rs
ResonateAndFire(ResonateAndFireNeuron),

// Voltage access
NeuronVariant::ResonateAndFire(n) => n.x,

// Factory
"ResonateAndFire" | "ResonateAndFireNeuron" => {
    Ok(NeuronVariant::ResonateAndFire(ResonateAndFireNeuron::new()))
}
```

---

## Performance Benchmarks

### Rust (Criterion 0.8)

Measured on i5-11600K @ 3.90 GHz, single-threaded, 2026-04-05.

| Benchmark | Iterations | Median | Per-step | Notes |
|-----------|-----------|--------|----------|-------|
| `resonate_and_fire_10k_steps` | 10,000 | 541 µs | **54.1 ns** | Linear ODE + sqrt per step |

### Python

| Metric | Value |
|--------|-------|
| Isolation throughput | ~105K steps/s (~9.5 µs/step) |

### Speedup

| Metric | Python | Rust | Speedup |
|--------|--------|------|---------|
| Per-step latency | ~9,500 ns | 54.1 ns | **~176×** |

### Numerical Stability

| Test | Duration | Result |
|------|----------|--------|
| 20,000 steps at I=0.5 | 1 s sim time | All state finite |
| Strong drive I=5.0 | 10K steps | Spikes, bounded |
| Negative b=-0.5 | 10K steps | Faster decay, stable |

---

## Usage Examples

### Basic Resonance (Python)

```python
from sc_neurocore.neurons.models.resonate_and_fire import ResonateAndFireNeuron

neuron = ResonateAndFireNeuron()
spikes = []
for t in range(10000):
    spike = neuron.step(current=0.5)
    if spike:
        spikes.append(t)
print(f"Spikes: {len(spikes)}")
```

### Frequency-Selective Response

```python
import numpy as np
from sc_neurocore.neurons.models.resonate_and_fire import ResonateAndFireNeuron

# Drive at natural frequency omega=1.0
neuron = ResonateAndFireNeuron()
spikes_resonant = 0
for t in range(10000):
    I = 0.3 * np.sin(1.0 * t * 0.05)  # omega * t * dt
    spikes_resonant += neuron.step(I)

# Drive off-resonance
neuron2 = ResonateAndFireNeuron()
spikes_off = 0
for t in range(10000):
    I = 0.3 * np.sin(3.0 * t * 0.05)  # 3x omega
    spikes_off += neuron2.step(I)

print(f"On-resonance: {spikes_resonant}, Off-resonance: {spikes_off}")
# Resonant drive produces more spikes
```

### Subthreshold Spiral Trajectory

```python
from sc_neurocore.neurons.models.resonate_and_fire import ResonateAndFireNeuron

neuron = ResonateAndFireNeuron()
x_trace, y_trace = [], []
for _ in range(200):
    neuron.step(current=0.3)  # subthreshold
    x_trace.append(neuron.x)
    y_trace.append(neuron.y)
# Plot y vs x: inward spiral (damped oscillation)
```

### Rust Backend (via PyO3)

```python
from sc_neurocore_engine import ResonateAndFireNeuron as RustRAF

neuron = RustRAF()
spikes = sum(neuron.step(0.5) for _ in range(10000))
state = neuron.get_state()
print(f"Spikes: {spikes}, x={state['x']:.4f}, y={state['y']:.4f}")
```

---

## Test Coverage

### Python Tests (27 total)

**File:** `tests/test_model_resonate_and_fire.py`

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | Construction, binary output, 2 vars evolve, finite, reset |
| Resonance | 5 | Fires under drive, resonant > off-resonant, subthreshold spiral, frequency selectivity, amplitude growth |
| Dynamics | 5 | Spiral trajectory, damped decay, threshold crossing, phase preservation, reset clears |
| Parameters | 4 | b controls damping, omega controls frequency, threshold effect, dt stability |
| Performance | 2 | Isolation throughput, network throughput |
| Pipeline | 4 | Population, projection, network spikes, analysis |
| Stability | 2 | Extended run, extreme drive |

### Rust Tests (6 total)

| Test | What is verified |
|------|-----------------|
| `rnf_fires` | Fires under drive |
| `rnf_reset_clears_state` | x=0, y=0 after reset |
| `rnf_bounded` | State finite under drive |
| `rnf_nan_no_panic` | NaN input safe |
| `rnf_negative_no_crash` | Negative input stable |
| `rnf_subthreshold_oscillation` | Oscillates below threshold |

### Coverage Summary

| Category | Python | Rust | Total |
|----------|--------|------|-------|
| Construction/reset | 3 | 1 | 4 |
| Resonance/dynamics | 10 | 1 | 11 |
| Parameters | 4 | 0 | 4 |
| Numerical stability | 2 | 3 | 5 |
| Performance | 2 | 0 | 2 |
| Pipeline | 4 | 0 | 4 |
| **Total** | **27** | **6** | **33** |

---

## Citations

1. **Izhikevich, E. M.** (2001).
   Resonate-and-fire neurons.
   *Neural Networks*, 14(6-7), 883–894.
   DOI: [10.1016/S0893-6080(01)00078-8](https://doi.org/10.1016/S0893-6080(01)00078-8)

2. **Izhikevich, E. M.** (2007).
   *Dynamical Systems in Neuroscience: The Geometry of Excitability and Bursting.*
   MIT Press. Chapter 10: Resonate-and-fire neurons.

3. **Richardson, M. J. E., Brunel, N., & Hakim, V.** (2003).
   From subthreshold to firing-rate resonance.
   *Journal of Neurophysiology*, 89(5), 2538–2554.
   DOI: [10.1152/jn.00955.2002](https://doi.org/10.1152/jn.00955.2002)

4. **Hutcheon, B. & Yarom, Y.** (2000).
   Resonance, oscillation and the intrinsic frequency preferences of neurons.
   *Trends in Neurosciences*, 23(5), 216–222.
   DOI: [10.1016/S0166-2236(00)01547-2](https://doi.org/10.1016/S0166-2236(00)01547-2)

5. **Ermentrout, G. B. & Terman, D. H.** (2010).
   *Mathematical Foundations of Neuroscience.* Springer.
   Chapter 6: Subthreshold oscillations and resonance.

---

*SC-NeuroCore v3.14.0 — ANULUM / Fortis Studio*
*© 2020–2026 Miroslav Šotek. All rights reserved.*
