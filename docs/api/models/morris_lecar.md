# MorrisLecarNeuron

**Module:** `sc_neurocore.neurons.models.morris_lecar`
**Rust:** `sc_neurocore_engine::neurons::simple_spiking::MorrisLecarNeuron`
**Reference:** Morris, C. & Lecar, H. (1981)
**Publication:** *Voltage oscillations in the barnacle giant muscle fiber.* Biophysical Journal, 35(1), 193–213.
**Family:** Conductance-based (2D calcium-potassium oscillator)
**State variables:** `v` (membrane voltage, mV), `w` (potassium activation, dimensionless)

---

## 1. Mathematical Formalism

The Morris-Lecar model is a two-variable conductance-based neuron
derived from the electrophysiology of barnacle giant muscle fibre.
The complete system from Morris & Lecar (1981):

### 1.1 Membrane Equation

$$
C_m \frac{dV}{dt} = -I_{Ca} - I_K - I_L + I_{\text{ext}}
$$

### 1.2 Ionic Currents

**Calcium (instantaneous):**

$$
I_{Ca} = g_{Ca} \cdot m_\infty(V) \cdot (V - E_{Ca})
$$

**Potassium (delayed):**

$$
I_K = g_K \cdot w \cdot (V - E_K)
$$

**Leak:**

$$
I_L = g_L \cdot (V - E_L)
$$

### 1.3 Steady-State Activation Functions

$$
m_\infty(V) = \frac{1}{2}\left(1 + \tanh\!\left(\frac{V - V_1}{V_2}\right)\right)
$$

$$
w_\infty(V) = \frac{1}{2}\left(1 + \tanh\!\left(\frac{V - V_3}{V_4}\right)\right)
$$

### 1.4 Potassium Time Rate

$$
\lambda(V) = \phi \cdot \cosh\!\left(\frac{V - V_3}{2 V_4}\right)
$$

### 1.5 Potassium Gating ODE

$$
\frac{dw}{dt} = \lambda(V) \cdot (w_\infty(V) - w)
$$

### 1.6 Euler Integration (Implementation)

The baseline Euler implementation computes $m_\infty$, $w_\infty$, and
$\lambda$ from the old voltage, then updates V and w:

```
f_V(V, w, I_ext) =
  (-g_Ca * m_inf(V) * (V - E_Ca) - g_K * w * (V - E_K)
   -g_L * (V - E_L) + I_ext) / C_m

f_w(V, w) = lam(V) * (w_inf(V) - w)

m_inf(V) = 0.5 * (1 + tanh((V - V1) / V2))
w_inf(V) = 0.5 * (1 + tanh((V - V3) / V4))
lam(V)   = phi * cosh((V - V3) / (2 * V4))
```

The maintained production path advances these two ODEs with a
candidate-first fourth-order Runge-Kutta step. Python still exposes
`baseline_euler` for historical comparison and `rosenbrock` for the
linearly implicit solver path, but the default Python model, Rust engine,
Go service, Julia counterpart, and Rust safety surface use the RK4
conductance update. All runtime surfaces validate finite conductance
parameters, positive membrane capacitance, positive activation slopes and
timestep, and the potassium activation envelope `w in [0, 1]` before
integration. They reject non-finite runtime current and fail closed if the
`cosh` potassium-rate term overflows or any RK4 derivative/state candidate
becomes non-finite; the previous `(V, w)` state is preserved on rejection.

---

## 2. Theoretical Context

### 2.1 Background

Morris & Lecar (1981) developed this model to explain the voltage
oscillations observed in barnacle giant muscle fibre. The model was
originally concerned with calcium and potassium conductances in muscle,
but its mathematical structure has made it the canonical example for
analysing excitability types and bifurcation mechanisms in neuroscience.

### 2.2 Excitability Classification

The Morris-Lecar model can exhibit both **Type-I** and **Type-II**
excitability depending on parameter values, making it the primary
teaching and research tool for excitability classification:

**Type-I excitability** (SNLC bifurcation):
- Parameters: V3 = 2, V4 = 30, phi = 0.04
- Onset of oscillation at zero frequency (saddle-node on limit cycle)
- Continuous f-I curve: arbitrarily slow spiking near threshold
- No subthreshold oscillations

**Type-II excitability** (Hopf bifurcation):
- Parameters: V3 = 12, V4 = 17.4, phi = 1/15 (default)
- Onset at finite frequency (subcritical or supercritical Hopf)
- Discontinuous f-I curve: minimum frequency at onset
- Subthreshold oscillations near bifurcation

### 2.3 Nullcline Analysis

**V-nullcline** (dV/dt = 0):

$$
w = \frac{-g_{Ca} \cdot m_\infty(V) \cdot (V - E_{Ca}) - g_L(V - E_L) + I}{g_K(V - E_K)}
$$

This is an N-shaped curve (cubic-like) when plotted in the (V, w) plane.

**w-nullcline** (dw/dt = 0):

$$
w = w_\infty(V)
$$

This is a sigmoid. The intersection determines the fixed point.

### 2.4 Relation to Other Models

| Property | Morris-Lecar | FitzHugh-Nagumo | HH |
|----------|-------------|----------------|-----|
| Dimensions | 2 | 2 | 4 |
| V nonlinearity | tanh(V) sigmoids | v − v³/3 cubic | α/β rates |
| Currents | I_Ca, I_K, I_L | — | I_Na, I_K, I_L |
| Excitability | Type-I or II | Type-II only | Type-II |
| Biophysical | Yes (conductances) | No (qualitative) | Yes |
| tanh per step | 2 | 0 | 0 |
| cosh per step | 1 | 0 | 0 |

### 2.5 Ca as Instantaneous Variable

The calcium conductance uses the instantaneous activation $m_\infty(V)$
— no differential equation for $m$. This is valid because the calcium
channel activation time constant is much faster than both the membrane
time constant and the potassium kinetics. This reduction from 3 to 2
variables is what makes the model analytically tractable.

### 2.6 Bifurcation Analysis

The Morris-Lecar model exhibits a rich bifurcation structure as the
input current I is varied:

**Type-II (default parameters, Hopf):**
1. I < I_hopf: Stable fixed point on left branch, neuron is silent
2. I = I_hopf: Subcritical Hopf bifurcation, unstable limit cycle
   appears, then folds to create a stable limit cycle
3. I_hopf < I < I_block: Stable limit cycle, neuron fires regularly
4. I = I_block: Second Hopf bifurcation, limit cycle disappears
5. I > I_block: Stable fixed point on right branch, depolarisation block

**Type-I (V3=2, V4=30, phi=0.04, SNLC):**
1. I < I_snlc: Stable fixed point and saddle point (on the N-shaped
   V-nullcline middle branch)
2. I = I_snlc: Saddle-node on limit cycle — the fixed point and saddle
   collide and annihilate, leaving an invariant circle
3. I > I_snlc: Stable limit cycle with period → ∞ as I → I_snlc
   from above (arbitrarily slow spiking)

The SNLC mechanism is the defining feature of Type-I excitability
and produces a continuous f-I curve with square-root onset:
f ∝ √(I − I_snlc).

### 2.7 Canard Phenomena

Near the Hopf bifurcation (Type-II), the Morris-Lecar model can
exhibit canard-type behaviour where the trajectory follows the
repelling middle branch of the V-nullcline for an extended interval
before jumping away. This produces mixed-mode oscillations (small
amplitude oscillations interspersed with large spikes) in a narrow
parameter window.

### 2.8 Depolarisation Block

At high input current, the V-nullcline shifts upward, moving the fixed
point to the right branch. The potassium activation w saturates, and the
model enters a high-voltage quiescent state — depolarisation block.
This produces a non-monotonic f-I curve: firing rate increases with I,
peaks, then decreases back to zero.

---

## 3. Pipeline Position

```
sc_neurocore Pipeline
├── Python layer
│   └── sc_neurocore.neurons.models.morris_lecar.MorrisLecarNeuron
│       ├── step(current) → int {0, 1}
│       ├── reset() → None
│       ├── _m_inf(v), _w_inf(v), _lam(v) — helper functions
│       ├── Population(MorrisLecarNeuron, n=N)
│       ├── Network(pop, drive, monitor)
│       └── Analysis: spike_count(), firing_rate(), isi()
│
├── Rust engine
│   └── sc_neurocore_engine::neurons::simple_spiking::MorrisLecarNeuron
│       ├── new() → Self
│       ├── step(&mut self, current: f64) → i32
│       └── reset(&mut self)
│
├── PyO3 binding
│   └── sc_neurocore_engine.MorrisLecarNeuron (Python class)
│       ├── __init__()
│       ├── step(current) → int
│       ├── reset()
│       └── get_state() → dict {v, w}
│
└── Network runner
    └── NeuronVariant::MorrisLecar(MorrisLecarNeuron)
        ├── Wired in network_runner.rs:203
        ├── Voltage access: network_runner.rs:477
        └── Factory: "MorrisLecar" | "MorrisLecarNeuron" → new()
```

Numerical safety contract: candidate `(V, w)` updates are validated
before mutation in Python, Go, Julia, and Rust. Invalid conductance
configuration, non-finite external drive, non-finite potassium-rate
evaluation, or a candidate activation outside `[0, 1]` leaves the
previous state intact and returns or raises the surface-specific
fail-closed signal.

---

## 4. Features

### 4.1 Core Features

- **Two conductances + leak:** I_Ca (instantaneous), I_K (delayed), I_L
- **Switchable excitability type:** Type-I or Type-II via V3/V4/phi
- **tanh activation functions:** Smooth sigmoids, analytically tractable
- **Depolarisation block:** Non-monotonic f-I curve at high drive
- **Phase-plane tractable:** 2D system allows complete nullcline analysis
- **No reset mechanism:** Continuous oscillation via limit cycle

### 4.2 Supported Operations

| Operation | Python | Rust | PyO3 |
|-----------|--------|------|------|
| step(current) → spike | ✅ | ✅ | ✅ |
| reset() | ✅ | ✅ | ✅ |
| get_state() → dict | — | — | ✅ (v, w) |
| Population wrapping | ✅ | via NeuronVariant | — |
| Network integration | ✅ | ✅ | — |
| Spike analysis | ✅ | — | — |

### 4.3 Parameter Sensitivity

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `g_ca` ↑ | More Ca current → larger depolarisation | 2–8 mS/cm² |
| `g_k` ↑ | Greater repolarisation → narrower spikes | 4–16 mS/cm² |
| `phi` ↑ | Faster w kinetics → shorter interspike interval | 0.02–0.2 |
| `V3` | Shifts w activation → changes excitability type | -10 to 20 mV |
| `V4` | w slope → steepness of activation | 10–35 mV |
| `c_m` ↑ | Slower membrane → longer spike period | 10–40 µF/cm² |

---

## 5. Usage Examples

### 5.1 Basic Simulation (Python)

```python
from sc_neurocore.neurons.models.morris_lecar import MorrisLecarNeuron

neuron = MorrisLecarNeuron()
spikes = []
for t in range(50000):
    spike = neuron.step(current=200.0)  # needs high drive
    if spike:
        spikes.append(t)

print(f"Spike count: {len(spikes)}")
```

### 5.2 Type-I vs Type-II Comparison

```python
from sc_neurocore.neurons.models.morris_lecar import MorrisLecarNeuron

# Type-II (default parameters)
n2 = MorrisLecarNeuron()
s2 = sum(n2.step(200.0) for _ in range(10000))

# Type-I (SNLC parameters)
n1 = MorrisLecarNeuron(v3=2.0, v4=30.0, phi=0.04)
s1 = sum(n1.step(200.0) for _ in range(10000))

print(f"Type-II spikes: {s2}, Type-I spikes: {s1}")
```

### 5.3 f-I Curve Sweep

```python
from sc_neurocore.neurons.models.morris_lecar import MorrisLecarNeuron

for I in [0, 50, 100, 200, 500, 1000, 5000]:
    neuron = MorrisLecarNeuron()
    spikes = sum(neuron.step(float(I)) for _ in range(10000))
    rate = spikes / (10000 * 0.1 / 1000)  # Hz
    print(f"I={I:5d}: {spikes:3d} spikes, {rate:.1f} Hz")
```

### 5.4 Rust Backend (via PyO3)

```python
from sc_neurocore_engine import MorrisLecarNeuron as RustML

neuron = RustML()
spikes = sum(neuron.step(200.0) for _ in range(10000))
state = neuron.get_state()
print(f"Spikes: {spikes}, v={state['v']:.2f}, w={state['w']:.4f}")
```

### 5.5 Phase Plane Trajectory

```python
from sc_neurocore.neurons.models.morris_lecar import MorrisLecarNeuron

neuron = MorrisLecarNeuron()
v_trace, w_trace = [], []
for _ in range(10000):
    neuron.step(current=200.0)
    v_trace.append(neuron.v)
    w_trace.append(neuron.w)
# Plot w vs v — limit cycle visible in (V, w) plane
```

---

## 6. Technical Reference

### 6.1 Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -60.0 | mV | Membrane voltage (initial) |
| `w` | 0.0 | — | K activation gating variable (initial) |
| `c_m` | 20.0 | µF/cm² | Membrane capacitance |
| `g_ca` | 4.0 | mS/cm² | Ca maximal conductance |
| `g_k` | 8.0 | mS/cm² | K maximal conductance |
| `g_l` | 2.0 | mS/cm² | Leak conductance |
| `e_ca` | 120.0 | mV | Ca reversal potential |
| `e_k` | -84.0 | mV | K reversal potential |
| `e_l` | -60.0 | mV | Leak reversal potential |
| `v1` | -1.2 | mV | Ca activation half-point |
| `v2` | 18.0 | mV | Ca activation slope |
| `v3` | 12.0 | mV | K activation half-point |
| `v4` | 17.4 | mV | K activation slope |
| `phi` | 1/15 | — | K time-scale factor |
| `dt` | 0.1 | ms | Integration timestep |
| `v_threshold` | 0.0 | mV | Spike detection threshold |

### 6.2 Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `step` | `(current: f64) → i32` | 0 or 1 | Advance one timestep |
| `reset` | `() → ()` | — | Reset v to -60.0, w to 0.0 |
| `new` | `() → Self` | — | Rust constructor with defaults |
| `get_state` | `() → dict` | v, w | PyO3 only: state inspection |

### 6.3 Runtime Implementation Comparison

| Aspect | Python | Rust safety | Go service | Julia |
|--------|--------|-------------|------------|-------|
| m_inf | math.tanh | f64::tanh | math.Tanh | tanh |
| w_inf | math.tanh | f64::tanh | math.Tanh | tanh |
| lambda | math.cosh | f64::cosh | math.Cosh | cosh |
| Baseline integration | old-V Euler | old-V Euler | old-V Euler | old-V Euler |
| Invalid state | ValueError at construction/step | NaN fail-closed state | NaN fail-closed state | NaN fail-closed state |

The runtime surfaces are verified against the same one-step current-balance
invariant. Published parity claims should cite the exact verification run
or benchmark artefact used for that release.

### 6.4 NeuronVariant Wiring

```rust
// network_runner.rs:203
MorrisLecar(MorrisLecarNeuron),

// network_runner.rs:477 — voltage access
NeuronVariant::MorrisLecar(n) => n.v,

// network_runner.rs:923 — factory
"MorrisLecar" | "MorrisLecarNeuron" => {
    Ok(NeuronVariant::MorrisLecar(MorrisLecarNeuron::new()))
}
```

---

## 7. Performance Benchmarks

### 7.1 Current multi-backend local regression

The maintained RK4 path is covered by
`benchmarks/bench_model_morris_lecar.py`, which writes
`benchmarks/results/local_python_2026-06-17_morris_lecar_rk4.json`.
That artefact records Python, Rust, Go, and Julia timing medians plus
source hashes for the touched implementation and test files. The run is
labelled `local_regression_non_isolated` and
`production_speed_claim=false`; it is evidence that the same RK4 contract
is runnable across maintained backends, not a production throughput claim.

Measured local regression values from
`benchmarks/results/local_python_2026-06-17_morris_lecar_rk4.json`:

| Backend | Median ns/step | Min ns/step | Max ns/step | Spikes | Evidence |
|---------|---------------:|------------:|------------:|-------:|----------|
| Python | 25467.72258 | 25179.76967 | 26262.524385 | 476 | RK4 reference |
| Rust engine | 322.44675 | 313.577755 | 335.575095 | 476 | RK4 engine example |
| Go service mirror | 230.2 | 229.4 | 241.7 | 0 | deterministic RK4 mirror |
| Julia mirror | 159.14346 | 157.318495 | 161.061555 | 476 | RK4 mirror |
| Mojo mirror | 216.30613497109152 | 213.5709249705542 | 222.3385649267584 | 476 | RK4 mirror |

### 7.2 Historical Rust Criterion evidence

Measured on i5-11600K @ 3.90 GHz, single-threaded, 2026-04-05. This
pre-RK4 Criterion result is retained as historical context only.

| Benchmark | Iterations | Median | Per-step | Notes |
|-----------|-----------|--------|----------|-------|
| `morris_lecar_10k_steps` | 10,000 | 810 µs | **81.0 ns** | 2 tanh + 1 cosh per step |

### 7.3 Historical Python

Measured on same hardware, single-threaded, 2026-04-04.

| Metric | Value |
|--------|-------|
| Isolation throughput | ~141K steps/s (~7.1 µs/step) |

### 7.4 Historical speedup

| Metric | Python | Rust | Speedup |
|--------|--------|------|---------|
| Per-step latency | ~7,100 ns | 81.0 ns | **~88×** |

The 88× speedup (lower than FHN's 221×) reflects the transcendental
function cost: each step requires 2× tanh and 1× cosh, which limit
the speedup achievable through Rust compilation alone.

### 7.5 Numerical Stability

| Test | Duration | Result |
|------|----------|--------|
| 20,000 steps at I=200 | 2 s sim time | All state variables finite |
| High drive I=5000 | 200 steps | v finite |
| Negative drive I=-30 | 200 steps | v finite |

---

## 8. Test Coverage

### 8.1 Python Tests (42 total)

**File:** `tests/test_model_morris_lecar.py` (40 tests)

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | Construction, binary output, variables evolve, finite, reset |
| Activation | 4 | m_inf sigmoid, w_inf sigmoid, lambda positive, tanh boundaries |
| Dynamics | 6 | Fires under drive, subthreshold silence, Type-II onset, non-monotonic f-I, depolarisation block, rate increase |
| Equations | 4 | dV formula, dw formula, V-nullcline, w-nullcline |
| Parameters | 6 | g_ca sweep, g_k sweep, phi sweep, V3/V4 Type-I, c_m effect, dt stability |
| Excitability | 4 | Type-I SNLC, Type-II Hopf, frequency discontinuity, subthreshold oscillation |
| Performance | 2 | Isolation throughput, network throughput |
| Pipeline | 4 | Population, projection wiring, network spikes, spike analysis |
| Stability | 5 | Extended run, extreme drive, negative input, NaN detection, bounded voltage |

**File:** `tests/test_new_neurons.py` (2 tests)

| Test | What is verified |
|------|-----------------|
| `test_fires` | Fires under drive |
| `test_w_recovery` | w variable evolves |

### 8.2 Rust Tests (7 total)

**File:** `engine/src/neurons/simple_spiking.rs`

| Test | What is verified |
|------|-----------------|
| `ml_fires` | Fires at I=200 |
| `ml_silent_without_input` | v bounded at I=0 |
| `ml_reset_clears_state` | v=-60.0, w=0.0 after reset |
| `ml_moderate_input_stable` | v finite at moderate drive |
| `ml_w_bounded` | 0 ≤ w ≤ 1 (activation range) |
| `ml_nan_no_panic` | NaN input does not crash |
| `ml_negative_no_crash` | v finite at negative drive |

### 8.3 Coverage Summary

| Category | Python | Rust | Total |
|----------|--------|------|-------|
| Construction/reset | 3 | 1 | 4 |
| Activation functions | 4 | 0 | 4 |
| Dynamics/spiking | 10 | 2 | 12 |
| Equations/nullclines | 4 | 0 | 4 |
| Excitability types | 4 | 0 | 4 |
| Parameters | 6 | 0 | 6 |
| Numerical stability | 5 | 3 | 8 |
| Performance | 2 | 0 | 2 |
| Pipeline | 4 | 0 | 4 |
| **Total** | **42** | **7** | **49** |

---

## Historical Significance

The Morris-Lecar model occupies a unique position in computational
neuroscience. Originally developed to explain muscle fibre oscillations,
it became the standard pedagogical and research model for:

- **Excitability classification** (Rinzel & Ermentrout 1998): The ML
  model was the first to demonstrate both Type-I and Type-II
  excitability in a single model with different parameter regimes.
- **Bifurcation theory in neuroscience** (Izhikevich 2007): The
  two-dimensionality makes complete phase-plane analysis possible,
  making it the go-to example for saddle-node, Hopf, and homoclinic
  bifurcations.
- **Noise-driven dynamics** (Tateno & Pakdaman 2004): The ML model
  with additive noise has been used extensively to study coherence
  resonance and stochastic resonance in neural systems.
- **Network synchronisation** (Ermentrout & Kopell 1991): Coupled ML
  oscillators demonstrate frequency-locking, phase-locking, and
  desynchronisation phenomena relevant to cortical rhythms.

---

## Numerical Considerations

- **Transcendental functions:** Each step requires 2× tanh and 1× cosh.
  These are the dominant computational cost. The Rust backend computes
  these via LLVM-optimised libm, approximately 10× faster than Python's
  math.tanh/cosh.
- **dt = 0.1 ms:** Adequate for the smooth tanh dynamics. The model
  has no stiffness issues — the fastest timescale is set by C_m and the
  conductances, while the slowest is controlled by phi.
- **phi = 1/15:** The potassium variable w evolves 15× slower than the
  membrane. Very small phi (< 0.01) can make w essentially frozen,
  requiring long simulations for w to equilibrate.
- **No reset:** Like FHN, this is an oscillatory model. Spike detection
  is via upward threshold crossing of the continuous limit cycle.
- **Bounded dynamics:** The conductance-based structure with reversal
  potentials ensures voltage stays within [E_K, E_Ca] = [-84, 120] mV
  for physiological parameters. No explicit clamping needed.

---

## 9. Citations

1. **Morris, C. & Lecar, H.** (1981).
   Voltage oscillations in the barnacle giant muscle fiber.
   *Biophysical Journal*, 35(1), 193–213.
   DOI: [10.1016/S0006-3495(81)84782-0](https://doi.org/10.1016/S0006-3495(81)84782-0)

2. **Rinzel, J. & Ermentrout, G. B.** (1998).
   Analysis of neural excitability and oscillations.
   In *Methods in Neuronal Modeling*, Koch, C. & Segev, I. (Eds.), MIT Press, 251–291.

3. **Izhikevich, E. M.** (2007).
   *Dynamical Systems in Neuroscience: The Geometry of Excitability and Bursting.*
   MIT Press. Chapter 4: Two-dimensional systems.

4. **Ermentrout, G. B. & Terman, D. H.** (2010).
   *Mathematical Foundations of Neuroscience.* Springer.
   Chapter 3: The Morris-Lecar model.

5. **Tsumoto, K., Kitajima, H., Yoshinaga, T., Aihara, K., & Kawakami, H.** (2006).
   Bifurcations in Morris-Lecar neuron model.
   *Neurocomputing*, 69(4-6), 293–316.
   DOI: [10.1016/j.neucom.2005.03.006](https://doi.org/10.1016/j.neucom.2005.03.006)

6. **Prescott, S. A., De Koninck, Y., & Bhatt, D. H.** (2008).
   Biophysical basis for three distinct dynamical mechanisms of action potential initiation.
   *PLoS Computational Biology*, 4(10), e1000198.
   DOI: [10.1371/journal.pcbi.1000198](https://doi.org/10.1371/journal.pcbi.1000198)

7. **Hodgkin, A. L.** (1948).
   The local electric changes associated with repetitive action in a non-medullated axon.
   *Journal of Physiology*, 107(2), 165–181.
   (Original Type-I/Type-II excitability classification)

---

*SC-NeuroCore v3.14.0 — ANULUM / Fortis Studio*
*© 2020–2026 Miroslav Šotek. All rights reserved.*
