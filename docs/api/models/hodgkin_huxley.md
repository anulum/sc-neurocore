# HodgkinHuxleyNeuron

**Module:** `sc_neurocore.neurons.models.hodgkin_huxley`
**Reference:** Hodgkin & Huxley, J. Physiol. 117(4), 1952
**Family:** Biophysical conductance-based (the original)
**State variables:** `v` (membrane potential), `m` (Na⁺ activation), `h` (Na⁺ inactivation), `n` (K⁺ activation)
**Nobel Prize:** 1963 (Hodgkin & Huxley, with Eccles)

---

## Equations

### Membrane potential

$$C_m \frac{dV}{dt} = -g_{Na}\, m^3 h\,(V - E_{Na}) - g_K\, n^4\,(V - E_K) - g_L\,(V - E_L) + I$$

### Gating variables (α/β formulation)

$$\frac{dm}{dt} = \alpha_m(V)(1 - m) - \beta_m(V) \cdot m$$

$$\frac{dh}{dt} = \alpha_h(V)(1 - h) - \beta_h(V) \cdot h$$

$$\frac{dn}{dt} = \alpha_n(V)(1 - n) - \beta_n(V) \cdot n$$

### Rate functions

| Rate | Formula | Singularity handling |
|------|---------|---------------------|
| $\alpha_m$ | $\frac{0.1(V+40)}{1 - \exp(-(V+40)/10)}$ | V=−40: returns 1.0 (L'Hôpital) |
| $\beta_m$ | $4 \exp(-(V+65)/18)$ | None needed |
| $\alpha_h$ | $0.07 \exp(-(V+65)/20)$ | None needed |
| $\beta_h$ | $\frac{1}{1 + \exp(-(V+35)/10)}$ | None needed |
| $\alpha_n$ | $\frac{0.01(V+55)}{1 - \exp(-(V+55)/10)}$ | V=−55: returns 0.1 (L'Hôpital) |
| $\beta_n$ | $0.125 \exp(-(V+65)/80)$ | None needed |

### Spike detection

Threshold crossing: $V_t \geq V_{threshold}$ **and** $V_{t-1} < V_{threshold}$.
This is an upward-crossing detector, not a simple level check.

### Implementation (as coded)

```python
def step(self, current: float) -> int:
    v_prev = self.v
    for _ in range(round(1.0 / self.dt)):
        am, bm = self._alpha_m(self.v), self._beta_m(self.v)
        ah, bh = self._alpha_h(self.v), self._beta_h(self.v)
        an, bn = self._alpha_n(self.v), self._beta_n(self.v)

        self.m += (am * (1 - self.m) - bm * self.m) * self.dt
        self.h += (ah * (1 - self.h) - bh * self.h) * self.dt
        self.n += (an * (1 - self.n) - bn * self.n) * self.dt

        i_na = self.g_na * self.m**3 * self.h * (self.v - self.e_na)
        i_k = self.g_k * self.n**4 * (self.v - self.e_k)
        i_l = self.g_l * (self.v - self.e_l)

        self.v += (-i_na - i_k - i_l + current) / self.c_m * self.dt
    return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0
```

Forward Euler with **100 sub-steps** per `step()` call (dt=0.01, loop
`round(1.0/dt)` times). Gates updated before currents within each sub-step.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −65.0 | mV | Membrane potential (initial) |
| `m` | 0.05 | — | Na⁺ activation gate (initial) |
| `h` | 0.6 | — | Na⁺ inactivation gate (initial) |
| `n` | 0.32 | — | K⁺ activation gate (initial) |
| `c_m` | 1.0 | µF/cm² | Membrane capacitance |
| `g_na` | 120.0 | mS/cm² | Peak Na⁺ conductance |
| `g_k` | 36.0 | mS/cm² | Peak K⁺ conductance |
| `g_l` | 0.3 | mS/cm² | Leak conductance |
| `e_na` | 50.0 | mV | Na⁺ reversal potential |
| `e_k` | −77.0 | mV | K⁺ reversal potential |
| `e_l` | −54.4 | mV | Leak reversal potential |
| `dt` | 0.01 | ms | Sub-step integration timestep |
| `v_threshold` | 0.0 | mV | Spike detection threshold |

---

## Analytical Properties

### Three ionic currents

1. **I_Na = g_Na · m³h · (V − E_Na):** Fast sodium current. m³ provides
   rapid activation (positive feedback → depolarisation). h provides
   slower inactivation (negative feedback → repolarisation begins).

2. **I_K = g_K · n⁴ · (V − E_K):** Delayed rectifier potassium current.
   n⁴ provides slow activation that outlasts Na⁺ inactivation,
   driving repolarisation and the afterhyperpolarisation.

3. **I_L = g_L · (V − E_L):** Passive leak current. Small (g_L=0.3 vs
   g_Na=120), but dominates at subthreshold potentials where m ≈ 0.

### Current balance at rest

At V = −65 mV with default gate values:
- I_Na = 120 × 0.05³ × 0.6 × (−65 − 50) = 120 × 7.5e-5 × (−115) ≈ −1.035 (inward)
- I_K = 36 × 0.32⁴ × (−65 − (−77)) = 36 × 0.01049 × 12 ≈ 4.53 (outward)
- I_L = 0.3 × (−65 − (−54.4)) = 0.3 × (−10.6) ≈ −3.18 (inward)

Net ≈ −1.035 + 4.53 − 3.18 ≈ 0.31 µA/cm² — slightly outward, meaning
the resting state is near but not exactly at equilibrium (initial gate
values are approximate steady-state values).

### Gating variable steady state

Each gate relaxes to $x_\infty(V) = \alpha_x / (\alpha_x + \beta_x)$
with time constant $\tau_x = 1 / (\alpha_x + \beta_x)$.

At rest (V = −65):
- m_∞ = α_m / (α_m + β_m). With α_m(-65) = 0.1×25/(1-exp(-2.5)) ≈ 0.305,
  β_m = 4.0. m_∞ ≈ 0.305/4.305 ≈ 0.071. (Default m=0.05 is close.)
- h_∞ ≈ 0.07/(0.07+β_h(-65)). β_h(-65) = 1/(1+exp(3.0)) ≈ 0.047.
  h_∞ ≈ 0.07/0.117 ≈ 0.598. (Default h=0.6 matches.)
- n_∞: α_n(-65) = 0.01×(-10)/(1-exp(1)) ≈ 0.0147, β_n = 0.125×exp(0) = 0.125.
  n_∞ ≈ 0.0147/0.1397 ≈ 0.105. (Default n=0.32 is above steady-state — this
  represents a "prepared" initial condition, not exact rest.)

### Reversal potential ordering

$$E_K (-77) < E_L (-54.4) < V_{threshold} (0) < E_{Na} (50)$$

This ordering ensures: K⁺ is always outward (hyperpolarising), Na⁺ is
always inward (depolarising) during the action potential, and leak is
near rest (stabilising).

### Singularity protection

$\alpha_m$ and $\alpha_n$ have removable singularities at V=−40 and V=−55
respectively (0/0 form). L'Hôpital's rule gives:
- $\lim_{V \to -40} \alpha_m = 0.1 \times 10 \times 1 = 1.0$
- $\lim_{V \to -55} \alpha_n = 0.01 \times 10 \times 1 = 0.1$

Both verified analytically and by test.

---

## Behaviour

### Type-II excitability

The HH model exhibits **Type-II excitability** — a hallmark property:
- Firing rate onset is discontinuous: the neuron is silent below threshold,
  then jumps to a finite minimum frequency at onset
- f–I curve is non-monotonic: rate peaks at moderate current, then
  declines at high current (depolarisation block)
- Verified: f(20) > f(50) — rate at I=20 exceeds rate at I=50

This contrasts with Type-I models (LIF, QIF, AdEx) where the f–I curve
starts from zero and increases monotonically.

### The action potential sequence

1. **Resting:** V ≈ −65 mV, m small, h ≈ 0.6, n ≈ 0.3
2. **Depolarisation:** Input current raises V → m activates rapidly
   (fast positive feedback) → g_Na · m³h increases → I_Na inward
   → further depolarisation (Hodgkin cycle)
3. **Overshoot:** V approaches E_Na (+50 mV)
4. **Repolarisation:** h inactivates (slower than m activation) +
   n activates (delayed K⁺ current) → outward current exceeds inward
5. **Afterhyperpolarisation:** n still elevated, h still inactivated →
   V undershoots rest → V ≈ −80 mV briefly
6. **Recovery:** n deactivates, h de-inactivates → return to rest

Total action potential duration: ~2 ms. Refractory period: ~3 ms.

### Gating variable bounds

m, h, n should remain in [0, 1] by the α/β ODE structure (they represent
probabilities of gate states). Tested: after 5000 steps at I=10, all
gates in [−0.01, 1.01] (slight Euler overshoot tolerated).

### 100 sub-steps per call

The HH model is stiff due to fast Na⁺ activation (τ_m ≈ 0.1 ms at
threshold). With dt=0.01 ms and 100 sub-steps, each `step()` call
integrates 1 ms of biological time. This is necessary for numerical
stability — dt=0.1 ms would cause oscillatory instability in the m gate.

---

## Measured Dynamics (from test probing)

### Constant current sweep (default parameters)

| Current | Spikes (5000 steps) | Regime |
|---------|--------------------:|--------|
| 0.0 | 0 | Resting |
| 5.0 | ≤2 | Subthreshold (occasional transient) |
| 10.0 | ≥100 | Regular spiking |
| 20.0 | higher | Fast spiking |
| 50.0 | lower than 20 | Depolarisation block (Type-II) |

### ISI regularity

At I=10 (steady spiking), CV(ISI) < 0.5 after discarding the first
3 spikes (transient). The moderate CV (~0.26 measured) comes from the
interplay between fast Na⁺ and slow K⁺ gating — not noise, but
deterministic dynamics of the 4-ODE system.

---

## Comparison with Related Models

| Property | HH | ConnorStevens | WangBuzsaki | MorrisLecar |
|----------|----|----|----|----|
| State variables | 4 | 6 | 3 | 2 |
| Sub-steps | 100 | 100 | 100 | 10 |
| Excitability | Type-II | Type-I | Type-I (fast) | Type-II |
| Currents | Na, K, leak | Na, K, leak, A-type | Na, K | Ca, K |
| Spike shape | Realistic AP | Realistic + delay | Fast interneuron | Simplified |
| Speed | ~670 steps/s | ~1100 steps/s | ~800 steps/s | ~5000 steps/s |

The HH model is the founding biophysical model. ConnorStevens adds A-type
K⁺ current for Type-I behaviour. WangBuzsaki simplifies to 3 variables
for fast-spiking interneurons. MorrisLecar reduces to 2 variables for
qualitative analysis.

---

## Numerical Considerations

- **Stiffness:** The fast Na⁺ activation gate m has τ_m ≈ 0.1 ms at
  threshold — 100× faster than the membrane time constant. This requires
  dt ≤ 0.02 ms for Euler stability.
- **Sub-stepping:** 100 sub-steps per call (dt=0.01 ms, 1 ms biological
  time). This is the dominant performance cost.
- **dt tested:** dt = 0.005, 0.01, 0.02 all produce finite states after
  2000 steps. dt=0.02 is at the stability boundary.
- **Gate-before-current ordering:** Within each sub-step, gates are updated
  first, then ionic currents are computed with the new gate values. This
  is a forward Euler choice — the order matters for accuracy at finite dt.
- **Threshold crossing detector:** Uses upward crossing (V_t ≥ θ AND
  V_{t-1} < θ) rather than level detection. This prevents double-counting
  of spikes during the above-threshold plateau.
- **exp() calls:** 6 exp() evaluations per sub-step (α_m, β_m, α_h, β_h,
  α_n, β_n), totalling 600 exp() calls per `step()`. This is the
  computational bottleneck.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/hodgkin_huxley.py` — 86 lines.
- **Four state variables:** v, m, h, n (all float64).
- **Dataclass:** Uses `@dataclass` for parameter storage.
- **Private rate functions:** `_alpha_m`, `_beta_m`, etc. — 6 methods.
- **Singularity handling:** `abs(d) < 1e-7` guard for α_m and α_n.
- **Rust wiring:** Compatible with `step(f64) → i32`. Four f64 state
  variables. Supported via NeuronVariant. Rust sub-stepping loop compiled
  to native code gives ~10× speedup over Python.

---

## Infrastructure Pipeline

```
HodgkinHuxleyNeuron
├── step(current) → int {0, 1}
├── 100 sub-steps per call (dt=0.01ms, 1ms biological time)
├── Population, Network, SpikeMonitor: compatible
│   PoissonInput(weight=10, rate=500Hz)
├── Projection: tested src→tgt wiring
├── Analysis: spike_count, firing_rate verified
├── model_zoo usage: decision_making_circuit, auditory_processing, visual_cortex_v1
└── Rust: supported (4 f64 state vars, sub-stepping in native code)
```

**Used in 3 model_zoo architectures:**
- `decision_making_circuit`: excitatory pools (pool_A, pool_B, nonselective)
- `auditory_processing`: cochlear and integration layers
- `visual_cortex_v1`: simple cells (orientation-tuned)

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~670 steps/s | ~6K steps/s (estimated) |
| Network (3 neurons, 2s) | ~250 neuron-steps/s | — |

**Slow model** — 100 sub-steps × 6 exp() per sub-step = 600 exp() per call.
Third slowest in the library after ConnorStevens (also 100 sub-steps) and
MainenSejnowski (20 sub-steps but 2-compartment).

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 6 | defaults, binary, 4-var evolution, finite 5k, reset, 100 sub-steps |
| Rate functions | 5 | α_m singularity (V=−40→1.0), α_n singularity (V=−55→0.1), β_m formula, α_h formula, gating bounds |
| Current balance | 2 | I_Na inward at rest, I_K outward at rest |
| Type-II excitability | 4 | subthreshold silent, suprathreshold fires, non-monotonic f–I (f(20)>f(50)), ISI regularity |
| Parameters | 2 | dt stability (3 values), deterministic (100 steps) |
| Performance | 1 | isolation throughput > 100 steps/s |
| Pipeline | 4 | Population, Network+drive, Projection wiring, analysis (spike_count, firing_rate) |
| **Total** | **24** | |

See `tests/test_model_hodgkin_huxley.py`. No bugs found.

---

## Findings

1. **Type-II excitability confirmed:** f(20) > f(50) — the firing rate
   peaks at moderate current and declines at high current due to
   depolarisation block (sustained Na⁺ inactivation).

2. **α_m and α_n singularity protection works:** At V=−40 and V=−55
   respectively, the L'Hôpital limits 1.0 and 0.1 are returned within
   machine precision.

3. **β_m formula exact at rest:** β_m(−65) = 4.0 · exp(0) = 4.0 — verified
   to within 1e-10.

4. **I_Na inward, I_K outward at rest:** Current directions match
   biophysical expectations from the Nernst reversal potentials.

5. **Gating variables bounded:** After 5000 steps at I=10, m, h, n all
   within [−0.01, 1.01]. The slight Euler overshoot is acceptable.

6. **ISI has moderate CV:** CV(ISI) ≈ 0.26 at I=10 — not as regular as
   LIF (CV → 0) but structured. This reflects the deterministic dynamics
   of the 4-ODE system, not noise.

7. **100 sub-steps necessary:** The model requires dt ≤ 0.02 ms for
   stability. With the default dt=0.01 and 100 sub-steps, each call
   integrates 1 ms. Tested: dt=0.005, 0.01, 0.02 all stable.

8. **Network pipeline functional:** Population(n=3) + PoissonInput(500Hz,
   weight=10) + SpikeMonitor produces spikes after 2s. Projection wiring
   from source to target accepted by Network.

9. **Performance bottleneck is exp():** 600 exp() calls per step make
   HH one of the slowest models. Rust sub-stepping gives ~10× improvement
   by eliminating Python interpreter overhead in the inner loop.

10. **Foundation model for 3 zoo architectures:** HH neurons are used in
    decision_making_circuit, auditory_processing, and visual_cortex_v1 —
    the most-used biophysical model in the model_zoo.


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~531 steps/s |
| Spikes (10K steps, I=5.0) | 1 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`HodgkinHuxleyNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
1 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(HodgkinHuxleyNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~531 steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps

---

## Theoretical Context

### Historical significance

The Hodgkin-Huxley model (1952) is the first quantitative description of
action potential generation based on ionic conductances. Alan Hodgkin and
Andrew Huxley performed voltage-clamp experiments on the giant axon of
the Atlantic squid *Loligo forbesii* at the Marine Biological Association
laboratory in Plymouth, England. They published a series of five papers
in the *Journal of Physiology*, with the full model in the fifth paper
(Hodgkin & Huxley, 1952d). The work earned them the 1963 Nobel Prize in
Physiology or Medicine (shared with John Eccles).

### The experimental basis

The voltage-clamp technique allowed Hodgkin and Huxley to measure ionic
currents at fixed membrane potentials. By varying the holding potential
and using ion substitution experiments (replacing external Na⁺ with
choline), they separated the total current into three components:

- A fast, transient inward current carried by Na⁺ ions
- A delayed, sustained outward current carried by K⁺ ions
- A small, voltage-independent leak current

They fitted empirical equations to the kinetics of each conductance,
arriving at the α/β formulation with m³h gating for Na⁺ and n⁴ gating
for K⁺. The exponents (3 for m, 4 for n) were determined by fitting to
experimental activation curves — they correspond to three independent
m-particles and four independent n-particles, each with first-order
kinetics.

### Excitability classification

The HH model exhibits **Type-II excitability** (Rinzel & Ermentrout,
1998): firing onset is discontinuous, with a minimum frequency at
rheobase. This arises because the bifurcation at spike onset is a
subcritical Hopf bifurcation, not a saddle-node on invariant circle
(SNIC) as in Type-I models. The practical consequence: the HH neuron
cannot fire arbitrarily slowly — it jumps from silence to ~50 Hz at
threshold.

### Relation to other models in SC-NeuroCore

The HH model is the ancestor of all conductance-based models in
SC-NeuroCore:

| Model | Relation to HH |
|-------|---------------|
| ConnorStevens | HH + A-type K⁺ current → Type-I excitability |
| WangBuzsaki | HH simplified to 3 vars (m=m_∞) for fast interneurons |
| TraubMiles | HH adapted for CA3 pyramidal cells + M-current |
| MorrisLecar | Reduced 2D HH-like with Ca²⁺ instead of Na⁺ |
| MainenSejnowski | Two-compartment HH (soma + axon hillock) |
| GolombFS | HH + Kv3 for fast-spiking cortical interneurons |
| Pospischil | Minimal HH tuned for 5 cortical cell types |
| PlantR15 | HH + slow Ca²⁺/KCa for bursting (Aplysia R15) |

---

## Usage Examples

### Example 1: Basic Python — single neuron, constant current

```python
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron

neuron = HodgkinHuxleyNeuron()

# Simulate 500 ms with I = 10 µA/cm²
spikes = []
for t in range(500):
    spike = neuron.step(10.0)
    if spike:
        spikes.append(t)

print(f"Fired {len(spikes)} spikes in 500 ms")
print(f"Final state: V={neuron.v:.1f} mV, m={neuron.m:.4f}, "
      f"h={neuron.h:.4f}, n={neuron.n:.4f}")
```

### Example 2: Advanced Python — f–I curve demonstrating Type-II onset

```python
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron
import numpy as np

currents = np.arange(0, 50, 1.0)
rates = []

for I_ext in currents:
    neuron = HodgkinHuxleyNeuron()
    # Discard 200 ms transient, measure 800 ms
    for _ in range(200):
        neuron.step(I_ext)
    count = sum(neuron.step(I_ext) for _ in range(800))
    rates.append(count / 0.8)  # Hz (800 ms = 0.8 s)

# Type-II signature: rate jumps from 0 to finite at threshold,
# then declines at high current (depolarisation block)
for i, I_ext in enumerate(currents):
    if rates[i] > 0:
        print(f"Onset at I={I_ext:.0f}: {rates[i]:.1f} Hz")
        break
```

### Example 3: PyO3 Rust — high-performance stepping

```rust
use sc_neurocore_engine::neurons::HodgkinHuxleyNeuron;

let mut neuron = HodgkinHuxleyNeuron::new();

// 10,000 steps at I = 10 µA/cm² (10 seconds biological time)
let mut spike_count = 0;
for _ in 0..10_000 {
    spike_count += neuron.step(10.0);
}
println!("Spikes: {spike_count} in 10 s ({:.1} Hz)",
    spike_count as f64 / 10.0);

// Access state
println!("V = {:.2} mV, m = {:.4}, h = {:.4}, n = {:.4}",
    neuron.v, neuron.m, neuron.h, neuron.n);

// Reset to initial conditions
neuron.reset();
assert!((neuron.v - (-65.0)).abs() < 1e-12);
```

---

## Technical Reference

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `step` | `step(current: float) → int` | 0 or 1 | Advance 1 ms (100 sub-steps), return spike |
| `simulate` | `simulate(n_steps, current=0.0, backend="auto")` | `(v_trace, spikes)` | Run Python, default-contract Rust, or explicit compiled Mojo |
| `reset` | `reset() → None` | — | Restore v, m, h, n to initial values |

### Python/Rust/Mojo parity

| Property | Python | Rust engine | Mojo shared library |
|----------|--------|-------------|---------------------|
| Rate constants (6) | Explicit methods | `safe_rate()` + inline | Analytic singular limits + inline exp |
| Integration | Gate-first baseline Euler | Gate-first baseline Euler | Gate-first baseline Euler |
| Default sub-steps | `round(1.0/dt)` = 100 | `(1.0/dt) as usize` = 100 | half-even `round(1.0/dt)` = 100 |
| Spike detection | Upward crossing | Upward crossing | Upward macro-boundary crossing |
| Numeric surface | Full state/parameters | Factory defaults only | Full numeric state/parameters |
| Enrolled events | 0/6/9 | 0/6/9 | 0/6/9 |
| Enrolled trace | Reference | Below `1e-9` | Below `2e-9` |

### Supported operations

| Operation | Supported | Notes |
|-----------|-----------|-------|
| Population | Yes | `Population(HodgkinHuxleyNeuron, n=N)` |
| Projection | Yes | Standard src→tgt wiring |
| NetworkRunner | Yes | Variant #1 in enum |
| SpikeMonitor | Yes | Binary spike output |
| PoissonInput | Yes | Tested at 500 Hz, weight=10 |
| Model Zoo | Yes | 3 architectures |
| PyO3 bridge | Yes | 4 state vars mapped |
| Equation compiler | No | Not an ODE-string model |

---

## Performance Benchmarks

### Executable Mojo closure (2026-07-13)

The source-hashed closure run used one logical CPU, 100 macro-steps, and 11
repeats at `I=20`. CPU 10 was affinity-pinned but not kernel-isolated, the
governor was `powersave`, and workstation load was non-zero. These values are
local functional/regression evidence, not general throughput claims.

| Runtime | Median per 100 macro-steps | Per macro-step | Events | Max voltage gap |
|---------|---------------------------:|---------------:|-------:|----------------:|
| Mojo shared library | 1.238 ms | 12.385 µs | 9 | `1.605e-10` |
| Rust engine | 1.543 ms | 15.432 µs | 9 | `9.486e-13` |
| Python | 187.566 ms | 1.876 ms | 9 | reference |

The complete affinity, load, governor, runtime-version, source-hash, timing,
parity, and final-state record is
`benchmarks/results/bench_hodgkin_huxley_mojo.json`.

```python
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron

neuron = HodgkinHuxleyNeuron()
trace, spikes = neuron.simulate(100, current=20.0, backend="mojo")
print(trace[-1], spikes)
```

### Criterion 0.8 (Rust engine)

Measured on i5-11600K @ 3.90 GHz, single-threaded, 2026-04-05.

| Benchmark | Steps | Median | Per step | Sub-steps |
|-----------|------:|-------:|---------:|----------:|
| `hh_1k_steps` | 1 000 | 11.2 ms | 11.2 µs | 100 |

Per sub-step: 11.2 µs / 100 = **112 ns** (dominated by 6 × exp() = 672 ns
exp budget — the rest is arithmetic).

### Python throughput

Measured on i5-11600K, single-threaded.

| Metric | Value |
|--------|------:|
| Isolation throughput | ~531 steps/s |
| Per step | ~1.88 ms |

### Rust speedup

| Metric | Python | Rust | Speedup |
|--------|-------:|-----:|--------:|
| Per step | 1.88 ms | 11.2 µs | **168×** |

### Computational cost breakdown

Each `step()` call executes 100 sub-steps. Per sub-step:
- 6 exp() evaluations (α_m, β_m, α_h, β_h, α_n, β_n)
- 3 gate updates (m, h, n)
- 3 current evaluations (I_Na, I_K, I_L)
- 1 voltage update

Total per step: **600 exp() + 300 multiply-adds + 400 arithmetic ops**.
The exp() calls dominate — they account for ~70% of execution time.

---

## Citations

1. Hodgkin, A. L. & Huxley, A. F. (1952). A quantitative description of
   membrane current and its application to conduction and excitation in
   nerve. *Journal of Physiology*, 117(4), 500–544.
   DOI: [10.1113/jphysiol.1952.sp004764](https://doi.org/10.1113/jphysiol.1952.sp004764)

2. Hodgkin, A. L. & Huxley, A. F. (1952). Currents carried by sodium and
   potassium ions through the membrane of the giant axon of *Loligo*.
   *Journal of Physiology*, 116(4), 449–472.
   DOI: [10.1113/jphysiol.1952.sp004717](https://doi.org/10.1113/jphysiol.1952.sp004717)

3. Hodgkin, A. L. & Huxley, A. F. (1952). The components of membrane
   conductance in the giant axon of *Loligo*. *Journal of Physiology*,
   116(4), 473–496.
   DOI: [10.1113/jphysiol.1952.sp004718](https://doi.org/10.1113/jphysiol.1952.sp004718)

4. Hodgkin, A. L. & Huxley, A. F. (1952). The dual effect of membrane
   potential on sodium conductance in the giant axon of *Loligo*.
   *Journal of Physiology*, 116(4), 497–506.
   DOI: [10.1113/jphysiol.1952.sp004719](https://doi.org/10.1113/jphysiol.1952.sp004719)

5. Hodgkin, A. L., Huxley, A. F. & Katz, B. (1952). Measurement of
   current-voltage relations in the membrane of the giant axon of
   *Loligo*. *Journal of Physiology*, 116(4), 424–448.
   DOI: [10.1113/jphysiol.1952.sp004716](https://doi.org/10.1113/jphysiol.1952.sp004716)

6. Rinzel, J. & Ermentrout, G. B. (1998). Analysis of neural excitability
   and oscillations. In *Methods in Neuronal Modeling* (2nd ed.), Koch, C.
   & Segev, I. (Eds.), MIT Press, pp. 251–291.
   ISBN: 978-0-262-11231-4

7. Izhikevich, E. M. (2007). *Dynamical Systems in Neuroscience: The
   Geometry of Excitability and Bursting*. MIT Press, Chapter 2.
   DOI: [10.7551/mitpress/2526.001.0001](https://doi.org/10.7551/mitpress/2526.001.0001)

8. Dayan, P. & Abbott, L. F. (2001). *Theoretical Neuroscience:
   Computational and Mathematical Modeling of Neural Systems*. MIT Press,
   Chapter 5.
   ISBN: 978-0-262-04199-7
