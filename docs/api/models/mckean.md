# McKeanNeuron

**Module:** `sc_neurocore.neurons.models.mckean`
**Rust:** `sc_neurocore_engine::neurons::simple_spiking::McKeanNeuron`
**Reference:** McKean, H. P. (1970)
**Publication:** *Nagumo's equation.* Advances in Mathematics, 4(3), 209–223.
**Family:** 2D oscillator (piecewise-linear FitzHugh-Nagumo)
**State variables:** `v` (membrane-like, fast), `w` (recovery, slow)

---

## 1. Mathematical Formalism

The McKean model is a piecewise-linear simplification of the FitzHugh-
Nagumo system. The cubic v-nullcline is replaced by three linear
segments, enabling exact analytical solutions in each segment.

### 1.1 System of ODEs

$$
\frac{dv}{dt} = f(v) - w + I_{\text{ext}}
$$

$$
\frac{dw}{dt} = \varepsilon(v - \gamma w)
$$

### 1.2 Piecewise-Linear Nonlinearity

$$
f(v) = \begin{cases}
-v & v < a/2 \\
v - a & a/2 \leq v < (1 + a)/2 \\
1 - v & v \geq (1 + a)/2
\end{cases}
$$

With $a = 0.25$: breakpoints at $v = 0.125$ and $v = 0.625$.

### 1.3 Nullclines

**v-nullcline** ($dv/dt = 0$): $w = f(v) + I$
This is a piecewise-linear N-shaped curve — the direct analogue of
the FHN cubic nullcline, but with corners instead of smooth curvature.

**w-nullcline** ($dw/dt = 0$): $w = v / \gamma$
A straight line with slope $1/\gamma = 2$.

### 1.4 Euler Integration

Both Python and Rust use simultaneous forward Euler:

```
dv = (f(v) - w + I) * dt
dw = epsilon * (v - gamma * w) * dt
v += dv;  w += dw
```

Runtime surfaces reject non-finite current and fail closed if a
corrupted runtime state or state update becomes non-finite. Python
raises a numerical exception before mutation; the Rust engine, Julia,
Go, and Rust safety counterparts return no spike while preserving the
previous `(v, w)` state on rejected updates.

### 1.5 Spike Detection

Upward crossing of $v_{\text{peak}} = 0.8$. Note this uses a peak
threshold, not a hard reset — the model oscillates continuously
through the piecewise-linear limit cycle.

---

## 2. Theoretical Context

### 2.1 Background

McKean (1970) introduced this model as an analytically tractable
version of the FitzHugh-Nagumo system. By replacing the smooth cubic
$v - v^3/3$ with three linear segments, the model admits exact
closed-form solutions for spike shape, period, and threshold in each
linear region. This makes it the primary tool for rigorous mathematical
analysis of excitable systems.

### 2.2 Advantages over FHN

- **Exact solutions:** Within each linear segment, the system is a
  constant-coefficient linear ODE → exponential solutions
- **Explicit threshold:** The breakpoints $a/2$ and $(1+a)/2$ define
  precise boundaries between excitable and oscillatory regions
- **Phase plane geometry:** Nullclines have corners → bifurcations
  occur at specific, calculable parameter values
- **Canard analysis:** Piecewise-linear systems allow rigorous canard
  theory (Desroches et al. 2012) without exponentially small terms

### 2.3 Relation to FHN

| Property | McKean | FitzHugh-Nagumo |
|----------|--------|----------------|
| v nonlinearity | Piecewise linear f(v) | Cubic v - v³/3 |
| Analytically solvable | Yes (per segment) | No (except nullclines) |
| Nullcline shape | N-shaped with corners | N-shaped smooth |
| Dynamics | Identical qualitative | Identical qualitative |
| Bifurcation analysis | Exact | Approximate |
| Computational cost | Same (no exp) | Same (no exp) |

### 2.4 Excitability Analysis

The McKean model exhibits the same Type-II excitability as FHN:
- Below oscillatory band: stable fixed point on left branch of f(v)
- Within oscillatory band: limit cycle oscillation
- Above oscillatory band: stable fixed point on right branch (block)

The transition points can be computed exactly from the intersection
of the w-nullcline with the corners of f(v).

### 2.5 Parameter a Controls Nullcline Shape

The parameter $a$ determines the width of the middle (unstable) branch:
- $a = 0$: middle branch has zero width → integrator limit
- $a = 0.5$: symmetric N-shape
- $a = 1$: middle branch extends to full range → always excitable

The default $a = 0.25$ gives a narrow middle branch — the model is
excitable but requires significant input to cross the threshold region.

### 2.6 Singular Perturbation Theory

With $\varepsilon = 0.01$ (default), the system is a singularly
perturbed problem. The fast dynamics (v) jump between the left and
right branches of f(v), while the slow dynamics (w) drift along each
branch. The piecewise-linear structure makes Fenichel theory and
geometric singular perturbation analysis particularly clean.

### 2.7 Oscillation Period (Analytical)

For the McKean model, the oscillation period can be computed exactly.
In the singular limit $\varepsilon \to 0$, the period is determined
by the time spent on the left and right slow manifolds:

$$
T = \frac{1}{\varepsilon} \left( \int_{\text{left}} \frac{dw}{g(v_L(w), w)} + \int_{\text{right}} \frac{dw}{g(v_R(w), w)} \right)
$$

where $v_L(w)$ and $v_R(w)$ are the left and right branches of f(v)
expressed as functions of w. Since f is linear in each branch, these
integrals yield logarithmic expressions that can be evaluated in
closed form.

For default parameters ($a = 0.25$, $\varepsilon = 0.01$, $\gamma = 0.5$),
the period scales as $T \propto 1/\varepsilon$, giving approximately
628 timesteps per cycle (verified by test).

### 2.8 Threshold and Excitability

The excitability threshold can be computed exactly. A perturbation
crosses threshold when it pushes v past the corner at $v = a/2 = 0.125$.
Once past this point, the fast dynamics carry v rapidly to the right
branch (v > (1+a)/2 = 0.625), creating the spike. The critical
current for sustained oscillation is:

$$
I_{\text{crit}} = f(v^*) - \frac{v^*}{\gamma}
$$

where $v^*$ is the fixed point on the left branch. Below $I_{\text{crit}}$:
excitable (single spike with sufficient perturbation). Above: oscillatory.

### 2.9 Applications

The McKean model is used primarily as:

1. **Teaching tool:** The piecewise-linear structure allows students to
   compute phase portraits, periods, and bifurcation diagrams by hand.
2. **Rigorous proof target:** Mathematical theorems about excitable
   systems are often first proved for McKean, then generalised.
3. **Network analysis:** Coupled McKean oscillators with gap junctions
   (Coombes 2008) allow exact synchronisation analysis.
4. **Numerical benchmark:** The exact solutions provide ground truth
   for verifying numerical integration schemes.
5. **Mixed-mode oscillation studies:** Canard-type dynamics in
   piecewise-linear systems (Desroches et al. 2012).

### 2.10 Comparison with Other Piecewise-Linear Models

| Model | Dimensions | Segments | Application |
|-------|-----------|----------|-------------|
| McKean | 2 | 3 (N-shape) | Excitability analysis |
| Absolute IF | 1 | 2 (ramp + reset) | Minimal spiking |
| ELIF | 1 | 2 + exp | Spike initiation |
| Coombes (2008) | 2 | 3 + gap junctions | Network sync |
| Tonnelier (2003) | 2 | 3 (cubic approx) | Travelling waves |

The McKean model occupies the sweet spot: complex enough for
qualitative dynamics (oscillation, excitability, bifurcation) but
simple enough for closed-form analysis.

---

## 3. Pipeline Position

```
sc_neurocore Pipeline
├── Python layer
│   └── sc_neurocore.neurons.models.mckean.McKeanNeuron
│       ├── step(current) → int {0, 1}
│       ├── reset() → None
│       ├── _f(v) → float  (piecewise-linear function)
│       ├── Population(McKeanNeuron, n=N)
│       ├── Network, Analysis pipeline
│
├── Rust engine
│   └── sc_neurocore_engine::neurons::simple_spiking::McKeanNeuron
│       ├── new() → Self
│       ├── step(&mut self, current: f64) → i32
│       ├── f_v(&self, v: f64) → f64
│       └── reset(&mut self)
│
├── PyO3 binding
│   └── sc_neurocore_engine.McKeanNeuron
│       └── get_state() → {v, w}
│
└── Network runner
    └── NeuronVariant::McKean(McKeanNeuron)
        ├── Factory: "McKean" | "McKeanNeuron" → new()
        └── Voltage access via n.v
```

---

## 4. Features

### 4.1 Core Features

- **Piecewise-linear:** Three linear segments replace FHN cubic
- **Analytically tractable:** Exact solutions per segment
- **Oscillatory band:** Spiking in limited I range
- **No transcendental functions:** Pure arithmetic (fastest possible)
- **No reset:** Continuous limit cycle oscillation
- **Singular perturbation:** Clean fast-slow decomposition

### 4.2 Supported Operations

| Operation | Python | Rust | PyO3 |
|-----------|--------|------|------|
| step(current) → spike | ✅ | ✅ | ✅ |
| reset() | ✅ | ✅ | ✅ |
| get_state() → dict | — | — | ✅ (v, w) |
| Population wrapping | ✅ | via NeuronVariant | — |

### 4.3 Parameter Sensitivity

| Parameter | Effect | Default |
|-----------|--------|---------|
| `a` | Nullcline breakpoints, excitability | 0.25 |
| `epsilon` | Timescale separation | 0.01 (100:1) |
| `gamma` | w-nullcline slope (1/γ) | 0.5 |
| `v_peak` | Spike detection threshold | 0.8 |

---

## 5. Usage Examples

### 5.1 Basic Oscillation

```python
from sc_neurocore.neurons.models.mckean import McKeanNeuron

neuron = McKeanNeuron()
spikes = sum(neuron.step(0.3) for _ in range(10000))
print(f"Spikes: {spikes}")
```

### 5.2 Piecewise-Linear Shape

```python
from sc_neurocore.neurons.models.mckean import McKeanNeuron

neuron = McKeanNeuron()
v_trace = []
for _ in range(5000):
    neuron.step(current=0.3)
    v_trace.append(neuron.v)
# v shows characteristic sharp corners from piecewise f(v)
```

### 5.3 Parameter Sweep

```python
from sc_neurocore.neurons.models.mckean import McKeanNeuron

for a in [0.1, 0.25, 0.5, 0.75]:
    neuron = McKeanNeuron(a=a)
    spikes = sum(neuron.step(0.3) for _ in range(5000))
    print(f"a={a:.2f}: {spikes} spikes")
```

### 5.4 Oscillatory Band Detection

```python
from sc_neurocore.neurons.models.mckean import McKeanNeuron

for I in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]:
    neuron = McKeanNeuron()
    spikes = sum(neuron.step(I) for _ in range(10000))
    print(f"I={I:.1f}: {spikes:4d} spikes {'(oscillating)' if spikes > 5 else '(silent)'}")
# Reveals the oscillatory band boundaries
```

### 5.5 Epsilon Effect on Period

```python
from sc_neurocore.neurons.models.mckean import McKeanNeuron

for eps in [0.005, 0.01, 0.02, 0.05]:
    neuron = McKeanNeuron(epsilon=eps)
    spikes = sum(neuron.step(0.3) for _ in range(10000))
    period = 10000 / max(spikes, 1)
    print(f"eps={eps:.3f}: {spikes} spikes, period~{period:.0f} steps")
# Smaller epsilon → longer period (T ~ 1/epsilon)
```

### 5.6 Phase Plane Trajectory with Nullclines

```python
from sc_neurocore.neurons.models.mckean import McKeanNeuron
import numpy as np

neuron = McKeanNeuron()
v_trace, w_trace = [], []
for _ in range(5000):
    neuron.step(current=0.3)
    v_trace.append(neuron.v)
    w_trace.append(neuron.w)

# Compute nullclines for overlay
v_range = np.linspace(-0.5, 1.5, 200)
a = 0.25
f_nullcline = np.where(v_range < a/2, -v_range,
              np.where(v_range < (1+a)/2, v_range - a, 1 - v_range))
v_nullcline_w = f_nullcline + 0.3  # + I
w_nullcline_w = v_range / 0.5  # v / gamma

# Plot: trajectory loops around the piecewise N-shaped nullcline
```

### 5.7 Population Simulation

```python
from sc_neurocore.neurons.models.mckean import McKeanNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput

pop = Population(McKeanNeuron, n=20, label="mckean_pop")
drive = PoissonInput(n=20, rate_hz=300.0, weight=0.5, dt=0.001, seed=42)
mon = SpikeMonitor(pop)
net = Network(pop, drive, mon)
net.run(duration=1.0, dt=0.001, backend="python")
print(f"Total spikes: {mon.count}")
```

### 5.8 Rust Backend

```python
from sc_neurocore_engine import McKeanNeuron as RustMcK

neuron = RustMcK()
spikes = sum(neuron.step(0.3) for _ in range(10000))
state = neuron.get_state()
print(f"Spikes: {spikes}, v={state['v']:.4f}, w={state['w']:.4f}")
```

---

## 6. Technical Reference

### 6.1 Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | 0.0 | — | Fast variable (initial) |
| `w` | 0.0 | — | Slow recovery variable (initial) |
| `a` | 0.25 | — | Piecewise breakpoint parameter |
| `epsilon` | 0.01 | — | Timescale separation |
| `gamma` | 0.5 | — | w-nullcline slope |
| `v_peak` | 0.8 | — | Spike detection threshold |
| `dt` | 0.1 | ms | Integration timestep |

### 6.1.1 Validation contract

The Python model and acceleration mirrors reject non-finite `v`, `w`,
`v_peak`, and runtime input before state mutation.  The piecewise
breakpoint parameter `a` must be finite and satisfy `0 < a < 1`, preserving
the three ordered McKean branches at `a/2` and `(1+a)/2`.  The scale
parameters `epsilon`, `gamma`, and `dt` must be finite and strictly
positive.

### 6.2 Python/Rust Implementation Comparison

| Aspect | Python | Rust |
|--------|--------|------|
| Source | `mckean.py` | `simple_spiking.rs` |
| f(v) | 3-branch if/elif/else | 3-branch if/else |
| Integration | Simultaneous Euler | Simultaneous Euler (fixed 0255685) |
| Exp per step | 0 | 0 |
| **Parity** | **EXACT** (pure arithmetic, no RNG) | |

### 6.3 NeuronVariant Wiring

```rust
NeuronVariant::McKean(McKeanNeuron),
"McKean" | "McKeanNeuron" => Ok(NeuronVariant::McKean(McKeanNeuron::new()))
```

---

## 7. Performance Benchmarks

### 7.1 Rust (Criterion 0.8)

Measured on i5-11600K @ 3.90 GHz, single-threaded, 2026-04-05.

| Benchmark | Iterations | Median | Per-step |
|-----------|-----------|--------|----------|
| `mckean_10k_steps` | 10,000 | 282 µs | **28.2 ns** |

### 7.2 Python

| Metric | Value |
|--------|-------|
| Isolation throughput | ~280K steps/s (~3.6 µs/step) |

### 7.3 Speedup

| Metric | Python | Rust | Speedup |
|--------|--------|------|---------|
| Per-step | ~3,600 ns | 28.2 ns | **~128×** |

Pure arithmetic with branches — slightly slower than FHN (11.3 ns)
due to the branch predictions in the piecewise f(v).

### 7.4 Numerical Stability

| Test | Duration | Result |
|------|----------|--------|
| 20,000 steps at I=0.3 | 2 s sim time | v, w finite |
| dt=0.05, 0.1, 0.2 | 10K steps each | All stable |
| Extreme I=100 | 200 steps | v finite (enters right branch) |
| Negative I=-5 | 200 steps | v finite (stays on left branch) |

The piecewise-linear dynamics prevent unbounded growth: each branch
has a negative slope at the extremes ($-v$ for left, $1-v$ for right),
guaranteeing global boundedness without explicit clipping.

### 7.5 Comparison with Other 2D Models

| Model | Per-step (ns) | Exp/step | Speedup vs Python |
|-------|--------------|----------|-------------------|
| FHN | 11.3 | 0 | 221× |
| HR | 9.0 | 0 | 444× |
| **McKean** | **28.2** | **0** | **128×** |
| Morris-Lecar | 81.0 | 3 | 88× |
| TermanWang | 122.8 | 1 | 50× |

McKean is slower than FHN despite being "simpler" because the
piecewise branches introduce conditional jumps that break the CPU
pipeline. FHN's smooth cubic compiles to straight-line arithmetic.

---

## 8. Test Coverage

### 8.1 Python Tests (35 total)

**File:** `tests/test_model_mckean.py`

| Category | Tests |
|----------|------:|
| Isolation | 6 |
| Piecewise f(v) | 5 |
| Oscillatory band | 5 |
| Analytical | 5 |
| Parameters | 5 |
| Singular perturbation | 3 |
| Performance | 2 |
| Pipeline | 4 |
| Validation | 34 |

### 8.2 Rust Tests (5 total)

| Test | What is verified |
|------|-----------------|
| `mckean_fires` | Fires under drive |
| `mckean_reset` | v=0, w=0 after reset |
| `mckean_bounded` | State finite |
| `mckean_nan` | NaN safe |
| `mckean_negative` | Negative I stable |

### 8.3 Summary

The module-specific Python suite covers the original behavioural and
pipeline contracts plus fail-closed validation of state, geometry,
timescale, and runtime input contracts. Rust/Go/Julia safety mirrors
now use the same McKean piecewise update rather than placeholder stubs.

---

## Numerical Considerations

- **No transcendental functions.** Pure arithmetic with branches.
  The piecewise f(v) uses only comparisons, additions, and one
  subtraction per evaluation.
- **Branch prediction cost.** The three-way if/else in f(v) can cause
  CPU pipeline stalls when v oscillates near breakpoints. This explains
  why McKean (28.2 ns) is slower than FHN (11.3 ns) despite having
  "simpler" dynamics — the smooth cubic is better for CPUs.
- **dt = 0.1:** Adequate for default ε = 0.01. The piecewise-linear
  dynamics have no stiffness — the fastest timescale is determined by
  the slope of f(v), which is at most 1 (on each branch).
- **ε controls minimum dt:** For very small ε (< 0.001), the fast
  jumps between branches happen in fewer timesteps, potentially
  missing threshold crossings. Use dt < ε × 10 as a rule of thumb.
- **Bounded dynamics.** All three branches of f(v) have negative slope
  at the extremes: f(v) → -v for v → -∞ and f(v) → 1-v for v → +∞.
  Combined with the linear w dynamics, this guarantees global
  boundedness of all trajectories.
- **Exact ground truth.** Within each linear segment, the analytical
  solution is $x(t) = e^{At}x(0) + A^{-1}(e^{At} - I)b$ where A is
  the 2×2 coefficient matrix and b is the constant term. This can be
  used to verify numerical accuracy of the Euler scheme.

---

## Historical Context

McKean's 1970 paper "Nagumo's equation" was published in *Advances in
Mathematics*, reflecting the model's contribution to pure mathematics
rather than neuroscience. The paper analysed the travelling wave
solutions of the Nagumo PDE (the spatially-extended version of FHN)
by replacing the cubic with a piecewise-linear function, allowing
explicit computation of wave speed and stability.

The model was later adopted by the computational neuroscience community
as a tool for:

- **Rinzel (1981):** Used the McKean model to develop the slow-fast
  analysis of neuronal bursting
- **Terman (1991):** Extended to networks for studying oscillator
  synchronisation
- **Coombes (2001, 2008):** Applied to networks with gap junctions,
  deriving exact synchronisation conditions
- **Tonnelier & Gerstner (2003):** Used for exact travelling wave
  analysis in neural field models

The model remains in active use in mathematical neuroscience because
closed-form solutions provide ground truth that numerical methods
cannot match.

---

## 9. Citations

1. **McKean, H. P.** (1970).
   Nagumo's equation.
   *Advances in Mathematics*, 4(3), 209–223.
   DOI: [10.1016/0001-8708(70)90023-X](https://doi.org/10.1016/0001-8708(70)90023-X)

2. **Desroches, M., Fernández-García, S., & Krupa, M.** (2012).
   Canards in piecewise-linear systems: explosions and super-explosions.
   *Proceedings of the Royal Society A*, 469(2154), 20120603.
   DOI: [10.1098/rspa.2012.0603](https://doi.org/10.1098/rspa.2012.0603)

3. **FitzHugh, R.** (1961).
   Impulses and physiological states in theoretical models of nerve membrane.
   *Biophysical Journal*, 1(6), 445–466.

4. **Rinzel, J. & Ermentrout, G. B.** (1998).
   Analysis of neural excitability and oscillations.
   In *Methods in Neuronal Modeling*, Koch & Segev (Eds.), MIT Press.

5. **Coombes, S.** (2008).
   Neuronal networks with gap junctions: A study of piecewise linear
   planar neuron models.
   *SIAM Journal on Applied Dynamical Systems*, 7(3), 1101–1129.
   DOI: [10.1137/070707579](https://doi.org/10.1137/070707579)

6. **Izhikevich, E. M.** (2007).
   *Dynamical Systems in Neuroscience.* MIT Press.
   Chapter 4: Piecewise-linear models.

7. **Tonnelier, A. & Gerstner, W.** (2003).
   Piecewise linear differential equations and integrate-and-fire neurons:
   insights from two-dimensional membrane models.
   *Neural Computation*, 15(7), 1621–1659.
   DOI: [10.1162/089976603321891845](https://doi.org/10.1162/089976603321891845)

---

*SC-NeuroCore v3.14.0 — ANULUM / Fortis Studio*
*© 2020–2026 Miroslav Šotek. All rights reserved.*
