# HindmarshRoseNeuron

**Module:** `sc_neurocore.neurons.models.hindmarsh_rose`
**Rust:** `sc_neurocore_engine::neurons::simple_spiking::HindmarshRoseNeuron`
**Reference:** Hindmarsh, J. L. & Rose, R. M. (1984)
**Publication:** *A model of neuronal bursting using three coupled first order differential equations.* Proceedings of the Royal Society B, 221(1222), 87–102.
**Family:** 3D oscillator / burster (chaotic)
**State variables:** `x` (fast, membrane-like), `y` (fast, recovery), `z` (slow, adaptation)

---

## 1. Mathematical Formalism

The Hindmarsh-Rose model is a three-variable phenomenological neuron
that captures the essential features of bursting dynamics. The complete
system from Hindmarsh & Rose (1984):

### 1.1 System of ODEs

$$
\frac{dx}{dt} = y - x^3 + bx^2 - z + I_{\text{ext}}
$$

$$
\frac{dy}{dt} = 1 - 5x^2 - y
$$

$$
\frac{dz}{dt} = r\bigl(s(x - x_r) - z\bigr)
$$

where $x$ is the membrane potential analogue, $y$ is the fast recovery
variable, $z$ is the slow adaptation variable, and $I_{\text{ext}}$ is
the external input current.

### 1.2 Fast Subsystem (x, y)

The fast subsystem (with $z$ frozen) is a 2D FitzHugh-Nagumo-like
system:

**x-nullcline** (dx/dt = 0): $y = x^3 - bx^2 + z - I$
**y-nullcline** (dy/dt = 0): $y = 1 - 5x^2$

The x-nullcline is a cubic curve whose shape depends on $z$ and $I$.
As $z$ varies slowly, the cubic shifts up and down, creating and
destroying limit cycles in the fast subsystem — the mechanism behind
bursting.

### 1.3 Slow Subsystem (z)

The slow variable $z$ acts as a negative feedback:
- During spiking: $x$ is large → $s(x - x_r) > z$ → $z$ increases
- $z$ increasing shifts the x-nullcline upward → eventually silences
  the fast subsystem (saddle-node bifurcation of cycles)
- During silence: $x \approx x_r$ → $z$ decreases slowly
- $z$ decreasing restores the limit cycle → bursting resumes

The parameter $r = 0.001$ sets the timescale separation: $z$ evolves
1000× slower than $x$ and $y$.

### 1.4 Spike Detection

Upward threshold crossing: spike when $x(t) \geq x_\theta$ and
$x(t - \Delta t) < x_\theta$, with $x_\theta = 1.0$.

### 1.5 Integration

The Python model defaults to fourth-order Runge-Kutta for the continuous
Hindmarsh-Rose ODE. The explicit Euler baseline remains available with
`integrator="euler"` for regression comparisons and cross-runtime current
balance checks.

Go, Rust safety, and Julia counterpart surfaces use the same candidate-first
fourth-order Runge-Kutta update as the default Python path:

```
k1 = f(x, y, z, I)
k2 = f(x + 0.5dt·k1x, y + 0.5dt·k1y, z + 0.5dt·k1z, I)
k3 = f(x + 0.5dt·k2x, y + 0.5dt·k2y, z + 0.5dt·k2z, I)
k4 = f(x + dt·k3x,     y + dt·k3y,     z + dt·k3z,     I)
state_next = state + dt/6 · (k1 + 2k2 + 2k3 + k4)
```

All three derivatives are computed from the old state before any
variable is updated. State is committed only after every RK4 stage and
the final candidate are finite.

All runtime surfaces reject non-finite input current and non-physical
time-step or slow-adaptation parameters before updating state. The Python,
Rust-engine, Go, Julia, and Mojo batch paths signal any non-finite RK4 stage;
the public Python dispatcher validates the complete trace and final state
before committing the instance. The scalar Python and Rust safety paths use
the same candidate-first rule, so the previous state is preserved on failure.

### 1.6 Schema and silicon contract

DOI `10.1098/rspb.1984.0024` supplies the three coupled ODEs and the canonical
`b=3`, `r=0.001`, `s=4`, and `x_rest=-1.6` operating parameters. The timestep,
classical RK4 discretisation, and upward `x_threshold=1.0` event are maintained
implementation choices rather than additional paper equations.

The paired TOML/JSON schemas use the same RK4/no-reset semantics as the hand
model. An earlier schema declared explicit Euler and an identity reset. That
identity reset disabled crossing-edge history in the schema runtime, so each
above-threshold timestep was incorrectly reported as a new event.

At Q16.16, hand model, TOML runner, JSON runner, and emitted RTL agree exactly
over 2,000 steps: `0/0/26/40/52` upward crossings at `I=0/2/3/4/5`. This is a
bounded crossing-count contract. Over 5,000 steps at the four bursting points
`I=2/3/4/5`, Q16.16 reports one additional crossing
(`10/49/86/115` versus float64 `9/48/85/114`), so long-window chaotic identity
is explicitly outside the claim.

The generated Q8.8 catalogue module has a port-only depth-4 SymbiYosys/Z3
bounded safety job. That proof covers reset/spike safety; it does not replace
the Q16.16 behavioural co-simulation evidence. The same committed RTL also
passes a real Yosys coarse-synthesis check, establishing bounded H2 evidence;
no timing, PPA, target-device, or physical-silicon claim follows from it.

---

## 2. Theoretical Context

### 2.1 Background

Hindmarsh & Rose (1984) developed this model to capture the bursting
behaviour observed in neurons that alternate between rapid spiking
and quiescent periods. The model extends the FitzHugh-Nagumo framework
by adding a third slow variable $z$ that modulates the fast dynamics,
producing the characteristic burst-silence pattern.

### 2.2 Three Dynamical Regimes

The model exhibits three qualitatively different regimes as $I$ varies:

1. **Quiescent** ($I \lesssim 2$): The system has a stable fixed point.
   $x$ remains near $x_r = -1.6$. No spikes.

2. **Bursting** ($2 \lesssim I \lesssim 5$): The system alternates
   between rapid spike bursts (fast oscillations in x, y) and silent
   inter-burst intervals (z slowly drifts). This is the signature
   behaviour of the HR model.

3. **Tonic spiking** ($I \gtrsim 5$): The external drive is large
   enough that the slow variable $z$ cannot suppress spiking. The
   model fires continuously.

### 2.3 Bursting Classification

The HR model produces **square-wave bursting** (also called fold/fold
or Type I bursting in Rinzel's classification). The mechanism:

1. Burst onset: fold bifurcation of the fast subsystem's fixed point
   → jump to the spiking branch
2. Burst termination: fold bifurcation of limit cycles in the fast
   subsystem → jump back to rest
3. The slow variable $z$ controls the transit between these bifurcations

### 2.4 Chaos in the HR Model

For specific parameter ranges (notably $r \approx 0.001$, $b \approx 3$,
$I \approx 3.25$), the HR model exhibits chaotic bursting:
- Irregular burst durations
- Non-periodic interburst intervals
- Positive Lyapunov exponent
- Strange attractor in (x, y, z) space

This makes the HR model one of the simplest systems that produces
biologically realistic chaos, and it has been used extensively in
studies of neural synchronisation and information coding.

### 2.5 Relation to Other Models

| Property | Hindmarsh-Rose | FitzHugh-Nagumo | Izhikevich |
|----------|---------------|----------------|------------|
| Dimensions | 3 (x, y, z) | 2 (v, w) | 2 (v, u) |
| Bursting | Yes (z variable) | No | Yes (reset rules) |
| Chaos | Yes (continuous) | No | Limited |
| Nonlinearity | x³, x² | v − v³/3 | v² |
| Reset | No (limit cycle) | No | Yes (hard) |
| Exp per step | 0 | 0 | 0 |

### 2.6 Homoclinic Bifurcation and Period-Adding

In the transition region between tonic spiking and bursting (I ≈ 3.3),
the HR model exhibits **period-adding cascades**: sequences of bursting
patterns where the number of spikes per burst increases by one at each
bifurcation. Between successive period-adding regimes lie windows of
chaotic behaviour. This is mediated by homoclinic orbits — trajectories
that approach the saddle fixed point arbitrarily closely before being
ejected back onto the spiking branch. The time spent near the saddle
determines the interburst interval, and its sensitivity to initial
conditions is the origin of chaos in the HR model.

### 2.7 Parameter Roles

- **$b = 3$:** Controls the shape of the cubic x-nullcline. Higher $b$
  makes the system more excitable (lower spiking threshold).
- **$r = 0.001$:** Timescale separation. Smaller $r$ → longer bursts
  and longer silences. $r = 0$ freezes $z$, reducing to 2D.
- **$s = 4$:** Coupling gain between fast and slow subsystems.
  Larger $s$ → greater adaptation → shorter bursts.
- **$x_r = -1.6$:** Resting potential analogue. The equilibrium
  of $z$ when $x = x_r$.

---

## 3. Pipeline Position

```
sc_neurocore Pipeline
├── Python layer
│   └── sc_neurocore.neurons.models.hindmarsh_rose.HindmarshRoseNeuron
│       ├── step(current) → int {0, 1}
│       ├── reset() → None
│       ├── Population(HindmarshRoseNeuron, n=N)
│       ├── Network(pop, drive, monitor)
│       └── Analysis: spike_count(), firing_rate(), isi()
│
├── Rust engine
│   └── sc_neurocore_engine::neurons::simple_spiking::HindmarshRoseNeuron
│       ├── new() → Self
│       ├── step(&mut self, current: f64) → i32
│       ├── simulate(n_steps, current) → (Vec<f64>, i64)   [RK4]
│       └── reset(&mut self)
│
├── PyO3 bindings
│   ├── sc_neurocore_engine.HindmarshRoseNeuron (Python class)
│   └── sc_neurocore_engine.py_hindmarsh_rose_simulate (N-step RK4)
│
├── Polyglot simulate chain (RK4): see "Polyglot acceleration" below
│   ├── Julia: src/sc_neurocore/accel/julia/neurons/hindmarsh_rose.jl
│   ├── Go:    src/sc_neurocore/accel/go/neurons/hindmarsh_rose/hindmarsh_rose.go (c-shared)
│   └── Mojo:  src/sc_neurocore/accel/mojo/neurons/hindmarsh_rose.mojo (FFI)
│
└── Network runner
    └── NeuronVariant::HindmarshRose(HindmarshRoseNeuron)
        ├── Wired in network_runner.rs:203
        ├── Voltage access: network_runner.rs:477 (n.x)
        └── Factory: "HindmarshRose" | "HindmarshRoseNeuron" → new()
```

---

## 4. Features

### 4.1 Core Features

- **Three dynamical regimes:** Quiescent, bursting, tonic spiking
- **Chaotic bursting:** Irregular burst patterns for specific parameters
- **Slow adaptation:** $z$ variable creates burst envelope on ~1000×
  slower timescale than fast spiking
- **Pure polynomial:** No transcendental functions — x³, x², multiplications
- **No reset:** Continuous dynamics via limit cycle, no artificial resets
- **Bounded orbits:** Cubic term ensures bounded x for physiological I

### 4.2 Supported Operations

| Operation | Python | Rust | PyO3 |
|-----------|--------|------|------|
| step(current) → spike | ✅ | ✅ | ✅ |
| reset() | ✅ | ✅ | ✅ |
| get_state() → dict | — | — | ✅ (x, y, z) |
| Population wrapping | ✅ | via NeuronVariant | — |
| Network integration | ✅ | ✅ | — |
| Spike analysis | ✅ | — | — |
| Failure-atomic batch | ✅ | ✅ | ✅ |

### 4.3 Parameter Sensitivity

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `b` ↑ | More excitable, lower spiking threshold | 2.5–4.0 |
| `r` ↓ | Longer bursts, longer silences | 0.0005–0.01 |
| `s` ↑ | Greater adaptation, shorter bursts | 2–6 |
| `x_rest` | Shifts z equilibrium, affects burst threshold | -2 to -1 |
| `I` | 0→2: silent, 2→5: bursting, >5: tonic | 0–10 |

---

## 5. Usage Examples

### 5.1 Basic Bursting (Python)

```python
from sc_neurocore.neurons.models.hindmarsh_rose import HindmarshRoseNeuron

neuron = HindmarshRoseNeuron()
spikes = []
for t in range(50000):
    spike = neuron.step(current=3.25)  # chaotic bursting regime
    if spike:
        spikes.append(t)

print(f"Spike count: {len(spikes)}")
# Compute ISIs to see burst structure
isis = [b-a for a,b in zip(spikes, spikes[1:])]
print(f"Mean ISI: {sum(isis)/len(isis):.1f}, Min: {min(isis)}, Max: {max(isis)}")
```

### 5.2 Three Regimes Sweep

```python
from sc_neurocore.neurons.models.hindmarsh_rose import HindmarshRoseNeuron

for I in [1.0, 3.0, 3.25, 5.0, 8.0]:
    neuron = HindmarshRoseNeuron()
    spikes = sum(neuron.step(I) for _ in range(20000))
    print(f"I={I:.2f}: {spikes:4d} spikes")
# Expect: ~0 (silent), burst, chaotic burst, burst/tonic, tonic
```

### 5.3 Slow Variable Trajectory

```python
from sc_neurocore.neurons.models.hindmarsh_rose import HindmarshRoseNeuron

neuron = HindmarshRoseNeuron()
z_trace, x_trace = [], []
for _ in range(50000):
    neuron.step(current=3.25)
    z_trace.append(neuron.z)
    x_trace.append(neuron.x)
# Plot z vs x: slow manifold with fast oscillations visible as loops
```

### 5.4 Rust Backend (via PyO3)

```python
from sc_neurocore_engine import HindmarshRoseNeuron as RustHR

neuron = RustHR()
spikes = sum(neuron.step(5.0) for _ in range(10000))
state = neuron.get_state()
print(f"Spikes: {spikes}")
print(f"x={state['x']:.3f}, y={state['y']:.3f}, z={state['z']:.3f}")
```

### 5.5 Burst Duration Analysis

```python
from sc_neurocore.neurons.models.hindmarsh_rose import HindmarshRoseNeuron

neuron = HindmarshRoseNeuron()
spikes = []
for t in range(100000):
    if neuron.step(current=3.0):
        spikes.append(t)

# Detect bursts: spikes within short ISI belong to same burst
bursts = []
current_burst = [spikes[0]]
for i in range(1, len(spikes)):
    if spikes[i] - spikes[i-1] < 50:  # within-burst ISI < 50 steps
        current_burst.append(spikes[i])
    else:
        bursts.append(current_burst)
        current_burst = [spikes[i]]
bursts.append(current_burst)
print(f"Bursts: {len(bursts)}, Mean spikes/burst: {sum(len(b) for b in bursts)/len(bursts):.1f}")
```

---

## 6. Technical Reference

### 6.1 Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `x` | -1.6 | — | Fast variable, membrane-like (initial) |
| `y` | -10.0 | — | Fast recovery variable (initial) |
| `z` | 2.0 | — | Slow adaptation variable (initial) |
| `b` | 3.0 | — | Quadratic coefficient (excitability) |
| `r` | 0.001 | — | Slow timescale (z dynamics rate) |
| `s` | 4.0 | — | Slow coupling strength |
| `x_rest` | -1.6 | — | Resting x value for z equilibrium |
| `dt` | 0.1 | ms | Integration timestep |
| `x_threshold` | 1.0 | — | Spike detection threshold |

### 6.2 Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `step` | `(current: f64) → i32` | 0 or 1 | Advance one timestep |
| `reset` | `() → ()` | — | Reset x=-1.6, y=-10.0, z=2.0 |
| `new` | `() → Self` | — | Rust constructor with defaults |
| `get_state` | `() → dict` | x, y, z | PyO3 only: state inspection |
| `simulate` | `(n_steps, current, backend) → (trace, spikes)` | x trace and events | Five-runtime failure-atomic RK4 batch |

### 6.3 Python/Rust Implementation Comparison

| Aspect | Python | Rust |
|--------|--------|------|
| Source | `hindmarsh_rose.py` (45 lines) | `engine/src/neurons/simple_spiking/hindmarsh_rose.rs` |
| Integration | RK4 default, Euler regression option | RK4 |
| Exp per step | 0 | 0 |
| Dependencies | None (pure arithmetic) | None (pure arithmetic) |
| **Parity** | **EXACT** (pure polynomial, no RNG) | |

### 6.4 NeuronVariant Wiring

```rust
// network_runner.rs:203
HindmarshRose(HindmarshRoseNeuron),

// network_runner.rs:477 — voltage access
NeuronVariant::HindmarshRose(n) => n.x,

// network_runner.rs:923 — factory
"HindmarshRose" | "HindmarshRoseNeuron" => {
    Ok(NeuronVariant::HindmarshRose(HindmarshRoseNeuron::new()))
}
```

---

## 7. Performance Benchmarks

### 7.1 Rust (Criterion 0.8)

Measured on i5-11600K @ 3.90 GHz, single-threaded, 2026-04-05.

| Benchmark | Iterations | Median | Per-step | Notes |
|-----------|-----------|--------|----------|-------|
| `hindmarsh_rose_10k_steps` | 10,000 | 90 µs | **9.0 ns** | Pure polynomial, 3 state vars |

### 7.2 Python

Measured on same hardware, single-threaded, 2026-04-04.

| Metric | Value |
|--------|-------|
| Isolation throughput | ~247K steps/s (~4.0 µs/step) |
| Spikes (10K steps, I=5.0) | 156 |

### 7.3 Speedup

| Metric | Python | Rust | Speedup |
|--------|--------|------|---------|
| Per-step latency | ~4,000 ns | 9.0 ns | **~444×** |

The 444× speedup — the highest among all neuron models — reflects the
pure polynomial nature: x³, x², and multiplications are the entire
computation. No transcendental functions, no branches, no table lookups.
The Rust compiler can fully vectorise and pipeline these operations.

### 7.4 Numerical Stability

| Test | Duration | Result |
|------|----------|--------|
| 20,000 steps at I=5.0 | 2 s sim time | All 3 state variables finite |
| 200 steps at I=5.0 (moderate) | 20 ms sim time | Bounded |
| Extended run (100K steps) | 10 s sim time | No divergence |

### 7.5 Polyglot acceleration

`step` runs one RK4 update, but `simulate(n_steps, current, backend=...)` is a
sequential recurrence (each step depends on the previous) that does not
vectorise — a compiled inner loop genuinely beats Python. The kernel carries a
full polyglot chain over the **RK4** integrator (the production default;
`simulate` raises for the `euler` integrator, which stays on `step()`):

```python
from sc_neurocore.neurons.models.hindmarsh_rose import HindmarshRoseNeuron

neuron = HindmarshRoseNeuron()                                    # integrator="rk4"
trace, spikes = neuron.simulate(2_000_000, current=3.0)           # auto -> Rust
trace, spikes = neuron.simulate(2_000_000, 3.0, backend="go")    # force a backend
```

`backend` accepts `"auto" | "rust" | "julia" | "go" | "mojo" | "python"`. `auto`
prefers Rust (it ships in the `sc_neurocore_engine` wheel). `trace[t]` is `x`
after step `t`; `spikes` counts upward crossings of `x_threshold`.

The RK4 right-hand side is **exact arithmetic** — the square and cube are written
`x*x` and `(x*x)*x` (bit-identical to Rust `x.powi(2)`/`x.powi(3)`, Julia
`x^2`/`x^3` and Go/Mojo `x*x`), with no transcendental functions. So even though
Hindmarsh-Rose is a three-dimensional **chaotic** burster, **Rust, Julia and Go
reproduce the NumPy trace bit-for-bit at every horizon** (verified over a
60,000-step chaotic run) — exactness is independent of the dynamics. This is the
sharp contrast with the chaotic *map* kernels (Cazelles, Medvedev), where a
clip/fold introduced a one-bit difference: here the polynomial flow has no such
step, so the bit-exact backends never diverge. Mojo's release build fuses some
RK4 multiply-adds into FMAs (≤8 ULP/step); the chaotic flow amplifies that ULP
into a growing whole-trace gap, so Mojo is validated on the per-step bound and
structural invariants, not whole-trace equality.

> Aligning the square/cube to `x*x` / `(x*x)*x` (from `x**2` / `x**3`) made the
> Python reference bit-identical to the engine's existing `x.powi(2)` /
> `x.powi(3)`, which is what lets the chain agree to the last bit.

#### Measured backends

Reproduce with `python benchmarks/bench_hindmarsh_rose_simulate.py --json
benchmarks/results/bench_hindmarsh_rose_simulate.json`. Workload: 2,000,000 RK4
steps, default parameters, current = 3.0 (bursting), median of 5 repeats.
**Non-isolated** (loaded workstation, Python 3.12 / NumPy 2.2.6) —
functional/regression evidence, not isolated-core release numbers.

| backend | median (ms) | speedup vs NumPy | parity Δ vs NumPy |
|---|---:|---:|---:|
| python (NumPy) | 4796.78 | 1.00× | 0 |
| mojo | 52.55 | 91.28× | 1.01e-08 (chaotic FMA amplification) |
| go | 70.40 | 68.14× | 0 (bit-exact) |
| julia | 89.72 | 53.47× | 0 (bit-exact) |
| rust | 82.22 | 58.34× | 0 (bit-exact) |

Mojo is fastest in raw throughput (the FMA contraction helps the cube-heavy RHS),
but because the chaotic flow amplifies its FMA ULP it is **not** chosen by
`auto`; `auto` selects Rust — the fastest backend that is both bit-exact and
ships in the wheel. The cube-heavy three-state RHS makes the NumPy reference the
slowest of the polynomial models, so the speedups are the highest (24–37×).

---

## 8. Test Coverage

### 8.1 Python Tests (30 total)

**File:** `tests/test_model_hindmarsh_rose.py` (28 tests)

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | Construction, binary output, 3 vars evolve, finite, reset |
| Dynamics | 6 | Fires under drive, bursting pattern, ISI bimodality, tonic spiking, subthreshold silence, z envelope |
| Equations | 3 | dx formula, dy formula, dz formula |
| Parameters | 4 | b excitability, r timescale, s coupling, regime transitions |
| Bursting | 4 | Burst detection, spikes per burst, interburst interval, burst regularity |
| Performance | 2 | Isolation throughput, network throughput |
| Pipeline | 4 | Population, projection wiring, network spikes, analysis |

**File:** `tests/test_new_neurons.py` (2 tests)

| Test | What is verified |
|------|-----------------|
| `test_fires` | Fires under drive |
| `test_z_adaptation` | z variable evolves |

### 8.2 Rust Tests (8 total)

**File:** `engine/src/neurons/simple_spiking/hindmarsh_rose.rs`

| Test | What is verified |
|------|-----------------|
| `default_matches_constructor_state` | `Default` and `new` start from the same x state |
| `simulate_matches_repeated_step` | Batched simulation is identical to repeated `step` calls |
| `hr_fires` | Fires under sustained drive |
| `hr_reset_clears_state` | x=-1.6, y=-10, z=2 after reset |
| `hr_moderate_input_stable` | All state finite at moderate drive |
| `hr_slow_z_evolves` | Slow adaptation state evolves |
| `hr_nan_no_panic` | NaN input does not crash |
| `hr_negative_no_crash` | State finite at negative I |

### 8.3 Coverage Summary

| Category | Python | Rust | Total |
|----------|--------|------|-------|
| Construction/reset | 3 | 2 | 5 |
| Dynamics/spiking | 6 | 3 | 9 |
| Equations | 3 | 0 | 3 |
| Bursting | 4 | 0 | 4 |
| Parameters | 4 | 0 | 4 |
| Numerical stability | 1 | 3 | 4 |
| Performance | 2 | 0 | 2 |
| Pipeline | 4 | 0 | 4 |
| **Total** | **30** | **8** | **38** |

---

## Historical Significance

The Hindmarsh-Rose model was introduced in two stages:

1. **1982 (Nature):** Hindmarsh & Rose published a 2D model (x, y only)
   as an improvement over the FitzHugh-Nagumo model, replacing the
   cubic v − v³/3 with x³ − bx² + y to better match the shape of
   real neuronal action potentials.

2. **1984 (Proc. R. Soc. B):** The critical addition of the slow
   variable z transformed the model into a burster. This was the first
   simple ODE model to reproduce the bursting patterns observed in
   molluscan neurons (Aplysia, Helix) with a biologically motivated
   mechanism (slow adaptation).

The HR model rapidly became the standard testbed for:

- **Dynamical systems analysis of bursting:** Rinzel (1987) used the HR
  model to develop the formal classification of bursting types (fold/fold,
  fold/Hopf, circle/fold, etc.) that remains the standard taxonomy.
- **Chaos in neural systems:** The model's chaotic regime demonstrated
  that deterministic chaos could arise from simple neural dynamics,
  challenging the assumption that neural irregularity requires noise.
- **Synchronisation studies:** Coupled HR neurons exhibit a rich
  repertoire of synchronisation states (in-phase, anti-phase, lag,
  chaotic synchronisation) relevant to understanding cortical rhythms.
- **Hardware implementations:** The purely polynomial dynamics make
  the HR model ideal for FPGA/ASIC neuromorphic hardware, requiring
  only multipliers and adders — no lookup tables for exp/tanh.

---

## Numerical Considerations

- **Pure polynomial dynamics:** x³, x², and multiplications only. No
  transcendental functions (exp, tanh, cosh). This makes the HR model
  the fastest per-step computation in the sc-neurocore library.
- **dt = 0.1:** Adequate for the default parameters. The fast subsystem
  (x, y) has eigenvalues that remain moderate for typical I values.
  For very small r (< 0.0001), longer simulations are needed to observe
  complete burst-silence cycles.
- **Simultaneous Euler:** Essential for this 3D system. Sequential
  update would create a different effective coupling between the fast
  and slow subsystems, altering burst duration and onset thresholds.
- **Bounded orbits:** The negative cubic term $-x^3$ dominates at
  large $|x|$, ensuring boundedness. No explicit clamping needed.
- **Stiffness:** Not stiff for default parameters. The timescale
  separation (r = 0.001) is moderate — adaptive methods are unnecessary.
- **Lyapunov sensitivity:** In the chaotic regime (I ≈ 3.25), tiny
  perturbations grow exponentially. Bit-exact parity between Python
  and Rust is maintained because both use identical arithmetic, but
  any change in dt or integration order will produce divergent
  trajectories after a few hundred steps.
- **FPGA suitability:** The absence of transcendental functions makes
  the HR model an ideal candidate for FPGA implementation. Only
  multipliers (for x³, x²) and adders are needed. Estimated resource
  usage: ~60 LUTs on a Xilinx Artix-7 for a single neuron instance.

---

## 9. Citations

1. **Hindmarsh, J. L. & Rose, R. M.** (1984).
   A model of neuronal bursting using three coupled first order differential equations.
   *Proceedings of the Royal Society B*, 221(1222), 87–102.
   DOI: [10.1098/rspb.1984.0024](https://doi.org/10.1098/rspb.1984.0024)

2. **Hindmarsh, J. L. & Rose, R. M.** (1982).
   A model of the nerve impulse using two first-order differential equations.
   *Nature*, 296(5853), 162–164.
   DOI: [10.1038/296162a0](https://doi.org/10.1038/296162a0)

3. **Rinzel, J.** (1987).
   A formal classification of bursting mechanisms in excitable systems.
   In *Mathematical Topics in Population Biology, Morphogenesis and Neurosciences*,
   Springer, 267–281.

4. **Izhikevich, E. M.** (2000).
   Neural excitability, spiking and bursting.
   *International Journal of Bifurcation and Chaos*, 10(6), 1171–1266.
   DOI: [10.1142/S0218127400000840](https://doi.org/10.1142/S0218127400000840)

5. **Barrio, R. & Shilnikov, A.** (2011).
   Parameter-sweeping techniques for temporal dynamics of neuronal systems:
   case study of Hindmarsh-Rose model.
   *Journal of Mathematical Neuroscience*, 1(1), 6.
   DOI: [10.1186/2190-8567-1-6](https://doi.org/10.1186/2190-8567-1-6)

6. **Storace, M., Linaro, D., & de Lange, E.** (2008).
   The Hindmarsh-Rose neuron model: bifurcation analysis and piecewise-linear
   approximations.
   *Chaos*, 18(3), 033128.
   DOI: [10.1063/1.2975967](https://doi.org/10.1063/1.2975967)

7. **González-Miranda, J. M.** (2007).
   Complex bifurcation structures in the Hindmarsh-Rose neuron model.
   *International Journal of Bifurcation and Chaos*, 17(9), 3071–3083.
   DOI: [10.1142/S0218127407018877](https://doi.org/10.1142/S0218127407018877)

---

*SC-NeuroCore v3.16.0 — ANULUM / Fortis Studio*
*© 2020–2026 Miroslav Šotek. All rights reserved.*
