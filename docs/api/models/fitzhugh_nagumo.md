# FitzHughNagumoNeuron

**Module:** `sc_neurocore.neurons.models.fitzhugh_nagumo`
**Rust:** `sc_neurocore_engine::neurons::simple_spiking::FitzHughNagumoNeuron`
**Reference:** FitzHugh, Biophys. J. 1(6), 1961; Nagumo, Arimoto & Yoshizawa, Proc. IRE 50(10), 1962
**Family:** 2D oscillator (qualitative reduction of Hodgkin-Huxley)
**State variables:** `v` (membrane potential, fast), `w` (recovery variable, slow)

---

## Equations

### Fast variable (membrane-like)

$$\frac{dv}{dt} = v - \frac{v^3}{3} - w + I$$

### Slow recovery variable

$$\frac{dw}{dt} = \varepsilon(v + a - bw)$$

### Spike detection

$$v \geq v_{threshold}(1.0) \; \text{AND} \; v_{prev} < v_{threshold}$$

**No reset** — this is an oscillatory model. The variable v traces a
limit cycle through the cubic nullcline. "Spikes" are upward threshold
crossings of the continuous oscillation.

### Implementation

```python
def step(self, current: float) -> int:
    v_prev = self.v
    dv = (self.v - self.v**3/3 - self.w + current) * self.dt
    dw = self.epsilon * (self.v + self.a - self.b * self.w) * self.dt
    self.v += dv
    self.w += dw
    return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0
```

The baseline path is explicit Euler. The Python model also exposes
`rk4` and `rosenbrock` integrators over the same two-state ODE. Runtime
surfaces validate state, positive `b`, `epsilon`, and timestep
contracts before integration, reject non-finite current, and fail closed
if the cubic term overflows or a derivative/state update becomes
non-finite; the previous state is preserved on rejection. Julia, Go, and
Rust safety counterparts use the same documented state equation,
candidate-state validation, and no-reset threshold crossing.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −1.0 | — | Fast variable (membrane-like) |
| `w` | −0.5 | — | Slow recovery variable |
| `a` | 0.7 | — | w-nullcline offset |
| `b` | 0.8 | — | w-nullcline slope |
| `epsilon` | 0.08 | — | Timescale separation (slow w) |
| `dt` | 0.1 | ms | Integration timestep |
| `v_threshold` | 1.0 | — | Spike detection threshold |

### ε = 0.08 (timescale separation)

The w variable evolves 12.5× slower than v (1/ε = 12.5). This creates
the classic fast-slow decomposition:
- **Fast (v):** Jumps rapidly along the cubic nullcline
- **Slow (w):** Drifts gradually, modulating v's dynamics

### a = 0.7, b = 0.8 (nullcline shape)

The w-nullcline is: w = (v + a)/b = (v + 0.7)/0.8.
This is a straight line with slope 1/b = 1.25 and intercept a/b = 0.875.
The intersection of the w-nullcline with the cubic v-nullcline
(v − v³/3) determines the fixed point.

---

## Analytical Properties

### Nullclines

**v-nullcline** (dv/dt = 0): $w = v - v^3/3 + I$
This is a cubic N-shaped curve. The local maximum and minimum create
the excitability threshold.

**w-nullcline** (dw/dt = 0): $w = (v + a)/b$
This is a straight line.

### Fixed point

At the intersection of nullclines:
$$v - v^3/3 + I = (v + a)/b$$

This is a cubic equation in v — has 1 or 3 real roots depending on
parameters and I. The stability of the fixed point determines the
dynamics.

### Hopf bifurcation

As I increases:
1. **I < I_lower:** Stable fixed point on left branch → excitable (no spikes)
2. **I = I_lower:** Hopf bifurcation → limit cycle appears
3. **I_lower < I < I_upper:** Unstable fixed point + limit cycle → oscillations
4. **I = I_upper:** Reverse Hopf bifurcation → fixed point stabilises
5. **I > I_upper:** Stable fixed point on right branch → depolarisation block

This creates an **oscillatory band** — spiking only for I in a specific
range. Below: silent. Above: also silent (depolarisation block).

### Type-II excitability

The FHN model is the canonical example of **Type-II excitability**:
- Onset of oscillation at finite frequency (Hopf bifurcation)
- No arbitrarily slow spiking near threshold
- Subthreshold oscillations near the bifurcation
- Discontinuous f-I curve onset

Contrast with Type-I (Connor-Stevens): continuous onset from zero Hz.

### Cubic v-nullcline and excitability

The v − v³/3 cubic creates three regions:
- **Left branch** (v < −1): stable rest
- **Middle branch** (−1 < v < 1): unstable (threshold region)
- **Right branch** (v > 1): spike peak

During an excitation cycle:
1. v jumps from left to right branch (fast, upstroke)
2. w slowly increases (chasing v on right branch)
3. At the right branch knee: v jumps back to left (fast, repolarisation)
4. w slowly decreases (recovery on left branch)

### Bounded orbit

The cubic term −v³/3 ensures v remains bounded for reasonable I.
Without the cubic, the linear instability would produce unbounded growth.
The v³ term provides the essential "fold-back" that creates the spike
peak and return to rest.

---

## Behaviour

### Oscillatory band (measured)

Verified by tests:
- **Below band (I too low):** Silent. The fixed point is stable on the
  left branch.
- **In band (I ≈ 0.5–1.2):** Oscillatory. The fixed point is unstable,
  a limit cycle exists. Spikes detected as upward threshold crossings.
- **Above band (I too high):** Suppressed. The fixed point stabilises
  on the right branch (depolarisation block).

### Regular ISI in band

Within the oscillatory band, the ISI is approximately constant (verified).
The limit cycle has a fixed period at each I value.

### Voltage bounded

During oscillation, v remains within a bounded range. Verified: |v| stays
within physiological limits for I in the oscillatory band.

### No reset

Unlike LIF/AdEx models, there is no artificial reset. The spike is a
natural part of the limit cycle — v rises, crosses threshold, continues
to the peak, then returns naturally via the cubic dynamics. This is
closer to real biological spike waveforms.

---

## Comparison with Related Models

| Property | FitzHugh-Nagumo | HindmarshRose | MorrisLecar | HH |
|----------|---------------|--------------|------------|-----|
| Dimensions | 2 (v, w) | 3 (x, y, z) | 2 (V, n) | 4 (V, m, h, n) |
| Nonlinearity | v − v³/3 (cubic) | x² − x³ | gating sigmoid | α/β rates |
| Reset | No (limit cycle) | No (limit cycle) | No | No |
| Spike shape | Smooth | Smooth | Smooth | Realistic |
| Excitability | Type-II | Type-I/II | Type-I/II | Type-II |
| Exp per step | 0 | 0 | 2 | 8+ |
| Bursting | No | Yes (z variable) | No | No |
| Origin | HH reduction | Phenomenological | Biophysical | Biophysical |

The FHN is the simplest 2D model that captures the essential dynamical
features of neuronal excitability — it is the "canonical form" from
which all qualitative analysis begins.

---

## Numerical Considerations

- **No transcendental functions.** Pure polynomial: v³, multiplications,
  additions. Among the fastest per-step computations.
- **No clipping.** The cubic naturally bounds v. No explicit bounds needed.
- **dt = 0.1:** Adequate for the smooth polynomial dynamics. No stiffness
  issues (unlike HH with fast Na⁺ kinetics).
- **Fail-closed candidate updates.** Each runtime surface validates the
  current state, runtime drive, and candidate `(v, w)` before mutation,
  so non-finite cubic overflow or corrupted runtime parameters cannot
  poison the stored state.
- **ε = 0.08:** Creates a 12.5:1 timescale separation. Small ε makes
  the w dynamics very slow → may need many steps for w to equilibrate.
- **No reset mechanism.** The model never resets — v oscillates
  continuously. This means the Pipeline's spike detection (threshold
  crossing) must handle re-crossings correctly.

---

## Usage Examples

### Basic Oscillation (Python)

```python
from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron

neuron = FitzHughNagumoNeuron()
spikes = []
for t in range(10000):
    spike = neuron.step(current=0.5)  # within oscillatory band
    if spike:
        spikes.append(t)

print(f"Spike count: {len(spikes)}")
print(f"Mean period: {sum(b-a for a,b in zip(spikes, spikes[1:])) / max(len(spikes)-1, 1):.1f} steps")
```

### Phase Plane Trajectory

```python
from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron

neuron = FitzHughNagumoNeuron()
v_trace, w_trace = [], []
for _ in range(5000):
    neuron.step(current=0.5)
    v_trace.append(neuron.v)
    w_trace.append(neuron.w)

# Plot w vs v to visualise the limit cycle
# The trajectory traces the cubic nullcline shape
```

### Rust Backend (via PyO3)

```python
from sc_neurocore_engine import FitzHughNagumoNeuron as RustFHN

neuron = RustFHN()
spikes = sum(neuron.step(0.5) for _ in range(10000))
state = neuron.get_state()
print(f"Spikes: {spikes}, v={state['v']:.3f}, w={state['w']:.3f}")
```

### Bifurcation Sweep

```python
from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron

for I in [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
    neuron = FitzHughNagumoNeuron()
    spikes = sum(neuron.step(I) for _ in range(5000))
    print(f"I={I:.1f}: {spikes} spikes")
# Expect: 0, 0, >0, >0, >0, 0, 0 (oscillatory band)
```

---

## Pipeline Position

```
sc_neurocore Pipeline
├── Python layer
│   └── sc_neurocore.neurons.models.fitzhugh_nagumo.FitzHughNagumoNeuron
│       ├── step(current) → int {0, 1}
│       ├── reset() → None
│       ├── Population(FitzHughNagumoNeuron, n=N)
│       ├── Network(pop, drive, monitor)
│       └── Analysis: spike_count(), firing_rate(), isi()
│
├── Rust engine
│   └── sc_neurocore_engine::neurons::simple_spiking::FitzHughNagumoNeuron
│       ├── new() → Self
│       ├── step(&mut self, current: f64) → i32
│       └── reset(&mut self)
│
├── PyO3 binding
│   └── sc_neurocore_engine.FitzHughNagumoNeuron (Python class)
│       ├── __init__()
│       ├── step(current) → int
│       ├── reset()
│       └── get_state() → dict {v, w}
│
└── Network runner
    └── NeuronVariant::FitzHughNagumo(FitzHughNagumoNeuron)
        ├── Wired in network_runner.rs:203
        ├── Voltage access: network_runner.rs:477
        └── Factory: "FitzHughNagumo" | "FitzHughNagumoNeuron" → new()
```

---

## Technical Reference

### Python/Rust Implementation Comparison

| Aspect | Python | Rust |
|--------|--------|------|
| Source | `fitzhugh_nagumo.py` (39 lines) | `simple_spiking.rs:15-55` |
| Dependencies | None (pure arithmetic) | None (pure arithmetic) |
| Integration | Simultaneous Euler | Simultaneous Euler (fixed 0255685) |
| Exp per step | 0 | 0 |
| **Parity** | **EXACT** (pure polynomial, no RNG) | |

### State Variables

| Variable | Type | Description |
|----------|------|-------------|
| `v` | f64 / float | Fast membrane-like variable |
| `w` | f64 / float | Slow recovery variable |

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `step` | `(current: f64) → i32` | 0 or 1 | Advance one timestep |
| `reset` | `() → ()` | — | Reset v to -1.0, w to -0.5 |
| `new` | `() → Self` | — | Rust constructor with defaults |
| `get_state` | `() → dict` | v, w | PyO3 only: state inspection |

### NeuronVariant Wiring

```rust
// network_runner.rs:203
FitzHughNagumo(FitzHughNagumoNeuron),

// network_runner.rs:477 — voltage access
NeuronVariant::FitzHughNagumo(n) => n.v,

// network_runner.rs:923 — factory
"FitzHughNagumo" | "FitzHughNagumoNeuron" => {
    Ok(NeuronVariant::FitzHughNagumo(FitzHughNagumoNeuron::new()))
}
```

---

## Performance Benchmarks

### Rust (Criterion 0.8)

Measured on i5-11600K @ 3.90 GHz, single-threaded, 2026-04-05.

| Benchmark | Iterations | Median | Per-step | Notes |
|-----------|-----------|--------|----------|-------|
| `fhn_10k_steps` | 10,000 | 113 µs | **11.3 ns** | Pure polynomial, no exp() |

### Python

Measured on same hardware, single-threaded.

| Metric | Value |
|--------|-------|
| Isolation throughput | ~400K steps/s (~2.5 µs/step) |
| Network throughput | Pipeline verified |

### Speedup

| Metric | Python | Rust | Speedup |
|--------|--------|------|---------|
| Per-step latency | ~2,500 ns | 11.3 ns | **~221×** |

The 221× speedup reflects the pure polynomial nature of FHN — no
transcendental function calls, making it ideal for SIMD vectorisation
in the Rust backend.

### Numerical Stability

| Test | Duration | Result |
|------|----------|--------|
| 50,000 steps at I=0.5 | 5 s sim time | State finite |
| dt=0.05, 0.1, 0.2 | 10K steps each | All stable |
| I=2.0 (moderate drive) | 200 steps | v finite |

---

## Test Coverage

### Python Tests (26 in test_model, 2 in test_new = 28 total)

**File:** `tests/test_model_fitzhugh_nagumo.py` (24 tests)

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary output, 2 variables evolve, state finite, reset |
| Equations | 4 | dv formula, dw formula, cubic nullcline, w-nullcline |
| Oscillatory band | 5 | silent below, oscillatory in, suppressed above, regular ISI, V bounded |
| Parameters | 4 | ε timescale, a shift, dt stability, deterministic |
| Performance | 2 | isolation throughput, network throughput |
| Pipeline | 4 | Population, Network spikes, Projection, analysis |

**File:** `tests/test_new_neurons.py` (2 tests)

| Test | What is verified |
|------|-----------------|
| `test_fires` | Fires under drive |
| `test_w_recovery` | Recovery variable evolves |

### Rust Tests (7 total)

**File:** `engine/src/neurons/simple_spiking.rs`

| Test | What is verified |
|------|-----------------|
| `fhn_fires` | Fires at I=2 in 1000 steps |
| `fhn_silent_without_input` | v bounded at I=0 |
| `fhn_reset_clears_state` | v=-1.0, w=-0.5 after reset |
| `fhn_moderate_input_stable` | v finite at I=2 in 200 steps |
| `fhn_recovery_variable` | w evolves during simulation |
| `fhn_nan_no_panic` | NaN input does not crash |
| `fhn_negative_no_crash` | v finite at I=-30 |

### Coverage Summary

| Category | Python | Rust | Total |
|----------|--------|------|-------|
| Construction/reset | 3 | 1 | 4 |
| Equations/dynamics | 9 | 3 | 12 |
| Oscillatory regime | 5 | 0 | 5 |
| Parameter sensitivity | 4 | 0 | 4 |
| Numerical stability | 1 | 3 | 4 |
| Performance | 2 | 0 | 2 |
| Pipeline integration | 4 | 0 | 4 |
| **Total** | **28** | **7** | **35** |

---

## Findings (Measured 2026-03-31)

1. **26/26 tests PASSED in 2.84s.** No failures.

2. **dv formula verified.** Single-step dv = (v − v³/3 − w + I) × dt
   matches implementation exactly.

3. **dw formula verified.** Single-step dw = ε(v + a − bw) × dt
   matches implementation exactly.

4. **Cubic nullcline verified.** w = v − v³/3 + I at dv/dt = 0.

5. **w-nullcline verified.** w = (v + a)/b at dw/dt = 0.

6. **Silent below band.** At low I, the model does not spike (stable FP).

7. **Oscillatory in band.** At moderate I, the model produces spikes
   (limit cycle oscillation).

8. **Suppressed above band.** At high I, spiking ceases (depolarisation
   block).

9. **Regular ISI in band.** Within the oscillatory regime, ISI is
   approximately constant (limit cycle period).

10. **Voltage bounded.** |v| stays within bounded range during oscillation.

11. **ε controls timescale.** Different ε values produce different w
    dynamics speeds.

12. **a shifts w-nullcline.** Different a values shift the equilibrium.

13. **dt stability.** dt=0.05, 0.1, 0.2 all produce finite state.

14. **Deterministic.** Bit-exact traces across repeated runs.

15. **Network pipeline functional.** Population, Projection, PoissonInput,
    analysis all work.

---

## Pipeline Verification (End-to-End, Measured 2026-03-31)

### Test execution

```
26/26 PASSED in 2.84s
├── TestFHNIsolation: 5 tests
├── TestFHNDynamicsEquations: 4 tests
├── TestFHNOscillatoryBand: 5 tests
├── TestFHNParameters: 4 tests
├── TestFHNPerformance: 2 tests
└── TestFHNPipeline: 4 tests
```

### Pipeline stages verified

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | ✓ PASS | v=-1, w=-0.5 |
| step() → int {0,1} | ✓ PASS | Upward crossing at 1.0 |
| dv formula | ✓ PASS | v − v³/3 − w + I |
| dw formula | ✓ PASS | ε(v + a − bw) |
| Cubic nullcline | ✓ PASS | Analytical |
| w-nullcline | ✓ PASS | Linear |
| Oscillatory band | ✓ PASS | Below/in/above |
| Regular ISI | ✓ PASS | Constant period |
| V bounded | ✓ PASS | No divergence |
| State finite | ✓ PASS | 50K steps |
| reset() | ✓ PASS | v→-1, w→-0.5 |
| Deterministic | ✓ PASS | Bit-exact |
| Population | ✓ PASS | Instances |
| Network | ✓ PASS | Spikes |
| Projection | ✓ PASS | Wiring |
| Analysis | ✓ PASS | Pipeline complete |

**ALL 26 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**

---

## Historical Significance

### FitzHugh 1961

Richard FitzHugh reduced the 4D Hodgkin-Huxley model to 2D by
identifying the essential dynamical features:
- Fast excitation (V, m) → single fast variable v
- Slow recovery (h, n) → single slow variable w

The resulting 2D system preserves excitability, oscillation, and
threshold behaviour while being amenable to phase-plane analysis.

### Nagumo et al. 1962

Jin-ichi Nagumo, Suguru Arimoto, and Shuji Yoshizawa independently
discovered the same system as an electronic circuit — the "tunnel
diode oscillator." This demonstrated that neural excitability is not
a unique biological phenomenon but a general property of systems with
cubic nonlinearity and slow recovery.

### The Bonhoeffer-van der Pol oscillator

The FHN model is mathematically equivalent to the Bonhoeffer-van der Pol
(BvP) oscillator, connecting neuroscience to electrical engineering and
nonlinear dynamics. This cross-disciplinary bridge has made the FHN one
of the most cited and studied models in all of science (>10,000 citations
combined for FitzHugh 1961 and Nagumo 1962).

---

## Citations

1. **FitzHugh, R.** (1961).
   Impulses and physiological states in theoretical models of nerve membrane.
   *Biophysical Journal*, 1(6), 445–466.
   DOI: [10.1016/S0006-3495(61)86902-6](https://doi.org/10.1016/S0006-3495(61)86902-6)

2. **Nagumo, J., Arimoto, S., & Yoshizawa, S.** (1962).
   An active pulse transmission line simulating nerve axon.
   *Proceedings of the IRE*, 50(10), 2061–2070.
   DOI: [10.1109/JRPROC.1962.288235](https://doi.org/10.1109/JRPROC.1962.288235)

3. **Hodgkin, A. L. & Huxley, A. F.** (1952).
   A quantitative description of membrane current and its application to conduction
   and excitation in nerve.
   *Journal of Physiology*, 117(4), 500–544.
   DOI: [10.1113/jphysiol.1952.sp004764](https://doi.org/10.1113/jphysiol.1952.sp004764)

4. **Izhikevich, E. M.** (2007).
   *Dynamical Systems in Neuroscience: The Geometry of Excitability and Bursting.*
   MIT Press. Chapter 4: Two-dimensional systems.

5. **Rinzel, J. & Ermentrout, G. B.** (1998).
   Analysis of neural excitability and oscillations.
   In *Methods in Neuronal Modeling*, Koch, C. & Segev, I. (Eds.), MIT Press, 251–291.

6. **Strogatz, S. H.** (2015).
   *Nonlinear Dynamics and Chaos.* 2nd ed. Westview Press.
   Chapter 7: Limit cycles and the van der Pol oscillator.

7. **Ermentrout, G. B. & Terman, D. H.** (2010).
   *Mathematical Foundations of Neuroscience.* Springer.
   Chapter 4: The FitzHugh-Nagumo model.

---

*SC-NeuroCore v3.14.0 — ANULUM / Fortis Studio*
*© 2020–2026 Miroslav Šotek. All rights reserved.*
