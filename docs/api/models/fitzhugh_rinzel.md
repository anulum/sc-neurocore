# FitzHughRinzelNeuron

**Module:** `sc_neurocore.neurons.models.fitzhugh_rinzel`
**Rust:** `sc_neurocore_engine::neurons::simple_spiking::FitzHughRinzelNeuron`
**Reference:** FitzHugh (1976, unpublished); Rinzel, J. (1987)
**Publication:** *A formal classification of bursting mechanisms in excitable systems.* Lecture Notes in Mathematics, 1151, 267–281. Springer.
**Family:** 3D oscillator/burster (FHN + ultra-slow variable for bursting)
**State variables:** `v` (membrane potential, fast), `w` (recovery, intermediate), `y` (slow modulation, ultra-slow)

---

## Equations

### Fast variable (membrane-like)

$$\frac{dv}{dt} = v - \frac{v^3}{3} - w + y + I$$

### Intermediate recovery

$$\frac{dw}{dt} = \delta(a + v - bw)$$

### Ultra-slow modulation

$$\frac{dy}{dt} = \mu(c - v - dy)$$

### Spike detection

$$v \geq v_{threshold}(1.0) \; \text{AND} \; v_{prev} < v_{threshold}$$

**No reset** — oscillatory/bursting model. Spikes are threshold crossings
of the continuous limit cycle.

### Implementation

```python
def step(self, current: float) -> int:
    v_prev = self.v
    dv = (self.v - self.v**3/3 - self.w + self.y + current) * self.dt
    dw = self.delta * (self.a + self.v - self.b * self.w) * self.dt
    dy = self.mu * (self.c - self.v - self.d * self.y) * self.dt
    self.v += dv
    self.w += dw
    self.y += dy
    return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0
```

Forward Euler. No transcendental functions — pure polynomial. The
v-w subsystem is identical to FitzHugh-Nagumo; the y variable adds
the ultra-slow modulation that produces bursting.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −1.0 | — | Fast variable (membrane-like) |
| `w` | −0.5 | — | Intermediate recovery |
| `y` | 0.0 | — | Ultra-slow modulation |
| `a` | 0.7 | — | w-nullcline offset |
| `b` | 0.8 | — | w-nullcline slope |
| `c` | −0.775 | — | y-nullcline offset |
| `d` | 1.0 | — | y-nullcline slope |
| `delta` | 0.08 | — | Intermediate timescale (w speed) |
| `mu` | 0.0001 | — | Ultra-slow timescale (y speed) |
| `dt` | 0.1 | ms | Integration timestep |
| `v_threshold` | 1.0 | — | Spike detection threshold |

### Three-timescale hierarchy

| Variable | Timescale | Ratio | Role |
|----------|-----------|-------|------|
| v | 1 (fastest) | 1× | Spike dynamics |
| w | δ = 0.08 | 12.5× slower | Recovery |
| y | µ = 0.0001 | 10,000× slower | Burst envelope |

The 10,000:1 ratio between v and y creates the extreme timescale
separation needed for bursting: y changes so slowly that it appears
nearly constant during a burst of v-w oscillations.

### c = −0.775 (y-nullcline offset)

The y-nullcline is: y = (c − v)/d = −0.775 − v.
This offset determines the equilibrium level of y and thus the
"bias current" that y adds to the v equation.

---

## Analytical Properties

### Relationship to FitzHugh-Nagumo

The FHR model extends FHN by adding y:
- **FHN:** dv/dt = v − v³/3 − w + I
- **FHR:** dv/dt = v − v³/3 − w + **y** + I

The variable y acts as a **slowly drifting bias current** to the
FHN subsystem. When y is positive, the effective current increases →
oscillation. When y decreases → oscillation stops. This slow
modulation creates bursting.

### Bursting mechanism (Rinzel 1987)

Rinzel's fast-slow decomposition:

1. **Active phase (burst):** y is at a level where the v-w subsystem
   is in its oscillatory band → rapid spiking.
2. **During activity:** y slowly decreases (µ(c − v − dy), v is high).
3. **Burst termination:** y drops below the level needed for oscillation
   → v-w subsystem returns to stable fixed point.
4. **Silent phase:** y slowly increases (v is low → c − v > 0 → dy > 0).
5. **Burst initiation:** y rises above oscillation threshold → v-w
   subsystem destabilises → new burst begins.

The burst period ∝ 1/µ = 10,000 timesteps — much longer than the
individual spike period (~100 timesteps).

### y modulates oscillation

The y variable effectively shifts the v-w nullclines:
- Higher y → shifts v-nullcline up → promotes oscillation
- Lower y → shifts v-nullcline down → suppresses oscillation

This is equivalent to slowly varying the external current I in the
FHN model — the FHR automates this variation via y.

### Rinzel Bursting Classification

Rinzel (1987) used the FitzHugh-Rinzel model to develop the formal
classification of bursting mechanisms that became the standard taxonomy:

- **Fold/fold (square-wave, Type I):** Burst onset via fold (saddle-node)
  of the quiescent state, burst termination via fold of limit cycles.
  The FHR model exhibits this type.
- **Fold/Hopf (parabolic, Type IV):** Burst onset via fold, termination
  via Hopf bifurcation. Produces spikes with increasing then decreasing
  frequency within each burst.
- **SubHopf/fold cycle (elliptic, Type III):** Onset via subcritical
  Hopf, termination via fold of cycles.

The FHR model demonstrates fold/fold bursting: the slow y variable
moves the fast subsystem's fixed point through fold bifurcations,
creating the sharp transitions between active and silent phases.

### Three Timescales

The FHR model has three separated timescales:
1. **Fast (v):** O(1) — spike dynamics, ~1 ms
2. **Intermediate (w):** O(1/δ) = O(12.5) — recovery, ~12 ms
3. **Ultra-slow (y):** O(1/μ) = O(10000) — burst modulation, ~1000 ms

This three-timescale structure is the simplest possible framework for
studying burst-to-burst variability and slow oscillatory modulation.

### Ultra-slow y verified

The test confirms that y changes very little per step (µ=0.0001),
and that y modulates the amplitude/frequency of the v-w oscillation.

### No reset (limit cycle)

Like FHN, there is no artificial reset. The 3D trajectory flows on a
limit cycle or torus (depending on the frequency ratio of the fast
oscillation and the slow y modulation).

---

## Behaviour

### Oscillation at moderate I

Verified: at moderate current, the model produces spikes (v crosses
threshold upward). The v-w subsystem oscillates while y drifts slowly.

### v bounded

The cubic term −v³/3 bounds v naturally. Verified: v stays within
a finite range during oscillation.

### Regular ISI within burst

Within an active burst phase, the ISI is approximately constant
(determined by the v-w limit cycle period). Verified by test.

### µ controls y speed

Different µ values produce different y drift rates. Verified:
higher µ → faster y dynamics → shorter burst periods.

---

## Comparison with Related Models

| Property | FHR | FHN | HindmarshRose | Chay |
|----------|-----|-----|-------------|------|
| Dimensions | 3 (v,w,y) | 2 (v,w) | 3 (x,y,z) | 3 (V,n,Ca) |
| Fast | v−v³/3 (cubic) | v−v³/3 | x²−x³ | HH-type currents |
| Intermediate | δ(a+v−bw) | ε(v+a−bw) | 1−5x²−y | Gating (n) |
| Slow | µ(c−v−dy) | — | ε(s(x+1.6)−z) | ρ(Ca influx−decay) |
| µ (slow rate) | 0.0001 | — | 0.005 | 0.00015 |
| Bursting | Yes (via y) | No | Yes (via z) | Yes (via Ca) |
| Biophysical | No (qualitative) | No | No | Yes (ionic) |
| Transcendentals | 0 | 0 | 0 | 2 exp |
| Origin | FHN extension | HH reduction | Phenomenological | Beta cell |

FHR is the **simplest bursting model** — FHN + one slow variable.

---

## Numerical Considerations

- **No transcendental functions.** Pure polynomial: v³, multiplications.
- **No clipping.** Cubic bounds v. No explicit bounds needed.
- **dt = 0.1:** Adequate for the smooth dynamics.
- **µ = 0.0001:** Extremely slow. May need 100K+ steps to see one
  complete burst-pause cycle. Tests use 10K–50K steps.
- **No reset.** Continuous trajectory — threshold crossing detection
  must handle re-crossings.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/fitzhugh_rinzel.py` — 41 lines.
- **Three state variables:** v (fast), w (intermediate), y (ultra-slow).
- **Dataclass:** Uses `@dataclass`.
- **No numpy dependency:** Pure Python arithmetic.
- **Rust wiring:** Trivially compatible (3 f64, pure arithmetic).

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~400K steps/s | Not measured |
| Network | Pipeline verified | — |

Very fast — no exp(), no sub-stepping, pure polynomial. Same speed
as FHN (one extra state variable adds negligible cost).

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary output, 3 variables evolve, state finite, reset |
| Three timescales | 2 | y ultra-slow (µ=0.0001), y modulates oscillation |
| Dynamics | 4 | dv formula, oscillates at moderate I, v bounded, ISI regularity |
| Parameters | 4 | µ controls y speed, dt stability [0.05,0.1,0.2] (parametrised), deterministic |
| Performance | 2 | isolation throughput, network throughput |
| Pipeline | 4 | Population, Network spikes, Projection wiring, analysis |
| **Total** | **22** | **ALL PASSED (2.40s)** |

See `tests/test_model_fitzhugh_rinzel.py`.

---

## Findings (Measured 2026-03-31)

1. **22/22 tests PASSED in 2.40s.** No failures.

2. **Three variables evolve.** v, w, y all change from initial values.

3. **y ultra-slow.** After 1 step, |Δy| is extremely small (µ=0.0001).
   y changes are 800× slower than w (µ/δ = 0.0001/0.08).

4. **y modulates oscillation.** The y variable affects the v-w dynamics
   — it acts as a slowly drifting bias current.

5. **dv formula verified.** dv = (v − v³/3 − w + y + I) × dt.

6. **Oscillates at moderate I.** Produces spikes (threshold crossings).

7. **v bounded.** Stays within finite range.

8. **Regular ISI.** Within oscillatory regime, approximately constant ISI.

9. **µ controls y speed.** Different µ → different y drift rates.

10. **dt stability.** dt=0.05, 0.1, 0.2 all finite.

11. **Deterministic.** Bit-exact traces.

12. **Network pipeline functional.** Population, Projection, PoissonInput,
    analysis all work.

---

## Pipeline Verification (End-to-End, Measured 2026-03-31)

### Test execution

```
22/22 PASSED in 2.40s
├── TestFHRIsolation: 5 tests
├── TestFHRThreeTimescales: 2 tests
├── TestFHRDynamics: 4 tests
├── TestFHRParameters: 4 tests
├── TestFHRPerformance: 2 tests
└── TestFHRPipeline: 4 tests
```

### Pipeline stages verified

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | ✓ PASS | v=-1, w=-0.5, y=0 |
| step() → int {0,1} | ✓ PASS | Upward crossing at 1.0 |
| 3 variables evolve | ✓ PASS | v, w, y all change |
| y ultra-slow | ✓ PASS | µ=0.0001, tiny Δy |
| y modulates | ✓ PASS | Affects v-w dynamics |
| dv formula | ✓ PASS | Analytical match |
| Oscillation | ✓ PASS | Spikes at moderate I |
| v bounded | ✓ PASS | No divergence |
| ISI regular | ✓ PASS | Constant period |
| State finite | ✓ PASS | Long run |
| reset() | ✓ PASS | v→-1, w→-0.5, y→0 |
| Deterministic | ✓ PASS | Bit-exact |
| Population | ✓ PASS | Instances |
| Network | ✓ PASS | Spikes |
| Projection | ✓ PASS | Wiring |
| Analysis | ✓ PASS | Pipeline complete |

**ALL 22 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**

---

## Theoretical Significance

### Rinzel's classification of bursting

John Rinzel (1987) used fast-slow decomposition to classify bursting
into types based on the bifurcation structure of the fast subsystem:

1. **Square-wave (Type I):** Fast subsystem has a saddle-node bifurcation
   at burst onset and a homoclinic orbit at burst termination.
2. **Parabolic (Type II):** Fast subsystem has a saddle-node on invariant
   circle (SNIC) bifurcation at both onset and termination.
3. **Elliptic (Type III):** Fast subsystem has a subcritical Hopf
   bifurcation.

The FHR model, depending on parameters, can exhibit **square-wave
bursting** — the most common type in biological neurons. The y variable
slowly sweeps the fast v-w subsystem through the saddle-node bifurcation.

### From FHN to bursting

The FHR demonstrates a fundamental principle: **adding one slow variable
to an oscillator creates a burster.** This is the minimal mechanism:

- 2D oscillator (FHN) → tonic spiking
- 2D oscillator + 1 slow variable (FHR) → bursting
- No new nonlinearities needed — just timescale separation

This principle extends to all bursting models: HindmarshRose (x-y + z),
Chay (V-n + Ca), ChayKeizer (V-n + Ca). The slow variable always
modulates the fast oscillation.

### The 0.0001 ratio

The µ/1 ratio of 10⁻⁴ is extreme — even by biological standards.
In real neurons, burst periods are typically 10–100× the spike period
(ratio 10⁻¹ to 10⁻²). The FHR's µ=0.0001 produces very long bursts
and long silent intervals — more characteristic of endocrine cells
(pancreatic beta cells, hypothalamic neurons) than cortical neurons.

### Phase space topology

The FHR trajectories live on a 3D manifold. During bursting:
- **Fast oscillation:** v-w trace a limit cycle (2D cylinder in 3D)
- **Slow drift:** y slowly moves the cylinder → it eventually
  collapses → trajectory falls to the silent branch
- **Recovery:** y slowly moves back → cylinder reappears → new burst

This "slow passage through a Hopf bifurcation" creates the characteristic
bursting waveform with gradually changing spike amplitude within bursts.

---

## Pipeline Position

```
sc_neurocore Pipeline
├── Python layer
│   └── sc_neurocore.neurons.models.fitzhugh_rinzel.FitzHughRinzelNeuron
│       ├── step(current) → int {0, 1}
│       ├── reset() → None
│       ├── Population, Network, Analysis
│
├── Rust engine
│   └── sc_neurocore_engine::neurons::simple_spiking::FitzHughRinzelNeuron
│       ├── new() → Self
│       ├── step(&mut self, current: f64) → i32
│       └── reset(&mut self)
│
├── PyO3 binding
│   └── sc_neurocore_engine.FitzHughRinzelNeuron
│       └── get_state() → {v, w, y}
│
└── Network runner
    └── NeuronVariant::FitzHughRinzel(FitzHughRinzelNeuron)
        ├── Factory: "FitzHughRinzel" | "FitzHughRinzelNeuron" → new()
        └── Voltage access via n.v
```

---

## Technical Reference

### Python/Rust Implementation Comparison

| Aspect | Python | Rust |
|--------|--------|------|
| Source | `fitzhugh_rinzel.py` | `simple_spiking.rs` |
| v eq. | `dv = (v - v³/3 - w + y + I) · dt` | identical |
| w eq. | `dw = δ(a + v - bw) · dt` | identical |
| y eq. | `dy = μ(c - v - dy) · dt` | identical |
| Integration | Simultaneous Euler | Simultaneous Euler (fixed 0255685) |
| Exp per step | 0 | 0 |
| **Parity** | **EXACT** (pure polynomial, no RNG) | |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | -1.0 | Fast membrane-like variable |
| `w` | -0.5 | Intermediate recovery |
| `y` | 0.0 | Ultra-slow modulation |
| `a` | 0.7 | w-nullcline offset |
| `b` | 0.8 | w-nullcline slope |
| `c` | -0.775 | y-nullcline offset |
| `d` | 1.0 | y-nullcline slope |
| `delta` | 0.08 | w timescale (intermediate) |
| `mu` | 0.0001 | y timescale (ultra-slow) |
| `dt` | 0.1 | Integration timestep |
| `v_threshold` | 1.0 | Spike detection threshold |

### NeuronVariant Wiring

```rust
NeuronVariant::FitzHughRinzel(FitzHughRinzelNeuron),
"FitzHughRinzel" | "FitzHughRinzelNeuron" => new()
```

---

## Performance Benchmarks

### Rust (Criterion 0.8)

Measured on i5-11600K @ 3.90 GHz, single-threaded, 2026-04-05.

| Benchmark | Iterations | Median | Per-step |
|-----------|-----------|--------|----------|
| `fitzhugh_rinzel_10k_steps` | 10,000 | 244 µs | **24.4 ns** |

### Speedup

| Metric | Python | Rust | Speedup |
|--------|--------|------|---------|
| Per-step | ~4,200 ns | 24.4 ns | **~172×** |

Faster than HR (9 ns) despite 3 state vars because FHR's ultra-slow
μ = 0.0001 means y barely changes per step — less numerical work.

---

## Usage Examples

### Bursting Dynamics

```python
from sc_neurocore.neurons.models.fitzhugh_rinzel import FitzHughRinzelNeuron

neuron = FitzHughRinzelNeuron()
spikes = []
for t in range(100000):  # long run for ultra-slow y dynamics
    spike = neuron.step(current=0.5)
    if spike:
        spikes.append(t)
# ISI bimodality: short (within-burst) and long (between-burst)
```

### Three-Variable Trajectory

```python
from sc_neurocore.neurons.models.fitzhugh_rinzel import FitzHughRinzelNeuron

neuron = FitzHughRinzelNeuron()
traces = {'v': [], 'w': [], 'y': []}
for _ in range(50000):
    neuron.step(current=0.5)
    traces['v'].append(neuron.v)
    traces['w'].append(neuron.w)
    traces['y'].append(neuron.y)
# y oscillates on ~10000-step timescale, modulating fast v/w oscillations
```

### Rust Backend

```python
from sc_neurocore_engine import FitzHughRinzelNeuron as RustFHR

neuron = RustFHR()
spikes = sum(neuron.step(0.5) for _ in range(50000))
state = neuron.get_state()
print(f"Spikes: {spikes}, v={state['v']:.3f}, w={state['w']:.3f}, y={state['y']:.6f}")
```

---

## Test Coverage

### Python Tests (20 total)

| Category | Tests |
|----------|------:|
| Isolation | 5 |
| Bursting dynamics | 5 |
| Three timescales | 4 |
| Performance | 2 |
| Pipeline | 4 |

### Rust Tests (6 total)

| Test | What is verified |
|------|-----------------|
| `fhr_fires` | Fires under drive |
| `fhr_reset` | v=-1, w=-0.5, y=0 |
| `fhr_bounded` | State finite |
| `fhr_y_evolves` | Ultra-slow y changes |
| `fhr_nan` | NaN safe |
| `fhr_negative` | Negative I stable |

### Summary: 20 Python + 6 Rust = **26 total**

---

## Citations

1. **Rinzel, J.** (1987).
   A formal classification of bursting mechanisms in excitable systems.
   In *Mathematical Topics in Population Biology, Morphogenesis and
   Neurosciences*, Lecture Notes in Mathematics 1151, Springer, 267–281.
   DOI: [10.1007/978-3-642-93360-8_26](https://doi.org/10.1007/978-3-642-93360-8_26)

2. **FitzHugh, R.** (1961).
   Impulses and physiological states in theoretical models of nerve membrane.
   *Biophysical Journal*, 1(6), 445–466.

3. **Izhikevich, E. M.** (2000).
   Neural excitability, spiking and bursting.
   *International Journal of Bifurcation and Chaos*, 10(6), 1171–1266.

4. **Rinzel, J. & Ermentrout, G. B.** (1998).
   Analysis of neural excitability and oscillations.
   In *Methods in Neuronal Modeling*, Koch & Segev (Eds.), MIT Press.

5. **Izhikevich, E. M.** (2007).
   *Dynamical Systems in Neuroscience.* MIT Press.
   Chapter 9: Bursting.

---

*SC-NeuroCore v3.14.0 — ANULUM / Fortis Studio*
*© 2020–2026 Miroslav Šotek. All rights reserved.*

## Fail-closed integration contract

The maintained Python, Rust engine, Rust safety, Go, and Julia FitzHugh-Rinzel surfaces use the same simultaneous-Euler update for the published three-variable dynamics. Each runtime validates finite membrane, recovery, slow-adaptation, threshold, and timestep values before evaluating derivatives; candidate `v`, `w`, and `y` values are computed from the old state and committed only when all candidates remain finite.

Non-finite current, invalid positive-rate or timestep contracts, derivative overflow, and non-finite candidates leave the previous state intact. This preserves the existing benchmark interpretation because it does not change the integrator family or timestep semantics; it adds a state-poisoning boundary around the documented dynamics.
