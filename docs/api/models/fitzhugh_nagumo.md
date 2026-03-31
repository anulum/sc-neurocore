# FitzHughNagumoNeuron

**Module:** `sc_neurocore.neurons.models.fitzhugh_nagumo`
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

Forward Euler. No transcendental functions — pure polynomial.

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
- **ε = 0.08:** Creates a 12.5:1 timescale separation. Small ε makes
  the w dynamics very slow → may need many steps for w to equilibrate.
- **No reset mechanism.** The model never resets — v oscillates
  continuously. This means the Pipeline's spike detection (threshold
  crossing) must handle re-crossings correctly.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/fitzhugh_nagumo.py` — 40 lines.
- **Two state variables:** v (fast), w (slow).
- **Dataclass:** Uses `@dataclass`.
- **No numpy dependency:** Pure Python arithmetic.
- **Simplest 2D ODE model** in SC-NeuroCore.
- **Rust wiring:** Trivially compatible (2 f64, pure arithmetic).

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~400K steps/s | Not measured |
| Network | Pipeline verified | — |

Very fast — no exp(), no sub-stepping, pure polynomial.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary output, 2 variables evolve, state finite, reset |
| Equations | 4 | dv formula verified, dw formula verified, cubic nullcline, w-nullcline |
| Oscillatory band | 5 | silent below band, oscillatory in band, suppressed above band, regular ISI, V bounded |
| Parameters | 4 | ε controls timescale, a shifts w-nullcline, dt stability [0.05,0.1,0.2], deterministic |
| Performance | 2 | isolation throughput, network throughput |
| Pipeline | 4 | Population, Network spikes, Projection wiring, analysis pipeline |
| **Total** | **26** | **ALL PASSED (2.84s)** |

See `tests/test_model_fitzhugh_nagumo.py`.

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
