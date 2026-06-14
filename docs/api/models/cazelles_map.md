# CazellesMapNeuron

**Module:** `sc_neurocore.neurons.models.cazelles_map`
**Reference:** Cazelles, Courbage & Rabinovich, Europhys. Lett. 56(4), 2001
**Family:** Map-based (discrete-time 2D bursting neuron)
**State variables:** `x` (fast variable, membrane-like), `y` (slow variable, recovery-like)

---

## Equations

### Fast map (logistic-like)

$$x_{n+1} = f(x_n) - y_n + I$$

### Slow recovery

$$y_{n+1} = y_n + \varepsilon \cdot (x_n - \sigma)$$

### Fast nonlinearity

$$f(x) = a \cdot x \cdot (1 - x)$$

This is the **logistic map** — the canonical example of deterministic chaos.
With a=3.8 (close to the chaotic regime of 3.57–4.0), the fast dynamics
are on the edge of chaos, producing complex spike patterns.

### Spike detection

$$x_n \geq x_{threshold}: \quad \text{return } 1$$

No reset — x evolves continuously. "Spikes" are threshold crossings of
the fast variable.

### Implementation

```python
def step(self, current: float) -> int:
    f = self.a * self.x * (1.0 - self.x)
    x_new = f - self.y + current
    y_new = self.y + self.epsilon * (self.x - self.sigma)
    self.x = min(2.0, max(-2.0, x_new))
    self.y = y_new
    return 1 if self.x >= self.x_threshold else 0
```

**Discrete-time map** — no ODE, no Euler, no dt. Each step() call is one
map iteration. This is fundamentally different from all ODE-based models:
there is no continuous time, only iteration count.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `x` | 0.1 | — | Fast variable (membrane-like) |
| `y` | 0.0 | — | Slow variable (recovery-like) |
| `a` | 3.8 | — | Logistic map parameter (nonlinearity strength) |
| `epsilon` | 0.01 | — | Slow timescale (y update speed) |
| `sigma` | 0.5 | — | Slow variable equilibrium point |
| `x_threshold` | 0.9 | — | Spike detection threshold |

### Logistic parameter a = 3.8

The logistic map f(x) = ax(1−x) has well-known dynamics:
- a < 1: x → 0 (extinction)
- 1 < a < 3: x → stable fixed point (1−1/a)
- 3 < a < 3.57: period-doubling cascade
- 3.57 < a < 4: chaotic regime (with periodic windows)
- a = 3.8: **near-chaotic** — complex dynamics with long transients

### Epsilon = 0.01 (slow timescale)

The y variable evolves 100× slower than x (ε=0.01). This creates the
timescale separation needed for bursting: x oscillates rapidly while y
drifts slowly, modulating the x dynamics.

### Sigma = 0.5 (equilibrium point)

y increases when x > σ=0.5 and decreases when x < 0.5. This creates
the slow negative feedback:
- During active phase (x high, >0.5): y increases → suppresses x
- During silent phase (x low, <0.5): y decreases → releases x

---

## Analytical Properties

### Map-based vs ODE-based models

| Property | Map (Cazelles) | ODE (HH, LIF) |
|----------|---------------|---------------|
| Time | Discrete (iterations) | Continuous (ms) |
| Integration | Exact (no Euler error) | Approximate (Euler/RK4) |
| dt | Not applicable | Required parameter |
| Stiffness | Not applicable | Can be problematic |
| Speed | Very fast (no exp) | Varies (exp, sub-steps) |
| Biological time | 1 iteration ≠ 1 ms | dt has physical meaning |

### Logistic map as spike generator

The logistic map f(x) = 3.8x(1−x) produces:
- f(0) = 0 (no activity → no activity)
- f(0.5) = 0.95 (moderate → near-threshold)
- f(0.9) = 0.342 (high → moderate, folding back)
- f(1.0) = 0 (maximum → zero, critical folding)

The folding at x=1 creates the "spike": x rises toward 1, then folds
back down — analogous to the Na⁺ activation → inactivation sequence
in HH-type models.

### Bursting mechanism

1. **y low:** x-map iterates freely → x can reach high values → "spikes"
2. **Each high x increases y:** ε(x − σ) > 0 when x > 0.5
3. **Rising y suppresses x:** x_{n+1} = f(x_n) − y → y acts as inhibition
4. **y reaches critical level:** x can no longer reach threshold → "silent"
5. **During silence:** x < σ=0.5 → y decreases slowly
6. **y drops enough:** x can again reach threshold → next burst
7. **Cycle repeats** with period ∝ 1/ε

### x clipped to [−2, 2]

The logistic map can produce x outside [0, 1] when combined with y and I.
The clip to [−2, 2] prevents divergence while allowing subthreshold
dynamics below 0.

### Chaotic bursting

Unlike periodic bursters (Chay, ChayKeizer), the Cazelles map can produce
**chaotic bursting:** bursts with irregular duration and inter-burst
intervals. The logistic map's near-chaotic dynamics (a=3.8) create complex,
non-repeating spike patterns within each burst.

---

## Behaviour

### Two-timescale dynamics

- **Fast (x):** Iterates of the logistic map — complex, potentially chaotic
- **Slow (y):** Drifts linearly with ε=0.01 — smooth, predictable

The fast variable x produces the spikes. The slow variable y modulates
whether x can spike. Together, they create bursting.

### Comparison with Rulkov map

| Property | Cazelles | Rulkov |
|----------|---------|--------|
| Fast map | Logistic (ax(1−x)) | Piecewise (α/(1+x²)) |
| Slow | y + ε(x − σ) | y − μ(x − σ) |
| Chaotic | Yes (a=3.8) | Yes (at some α) |
| Complexity | Minimal (2 lines) | Slightly more |
| Reference | Cazelles 2001 | Rulkov 2002 |

Both produce qualitatively similar bursting. Cazelles uses the logistic
map (more mathematically studied), Rulkov uses a custom piecewise map
(more biophysically motivated).

### No ODE, no dt, no exp

The model uses only:
- One multiplication (a × x)
- One subtraction (1 − x)
- One multiplication (× previous)
- One addition (− y + I)
- One multiplication (ε × (x − σ))
- One clip

**No transcendental functions.** This is the computationally cheapest
model in SC-NeuroCore (tied with TrueNorth and ThresholdLinearRate).

---

## Pipeline Verification (End-to-End, Measured 2026-03-31)

### Test execution

```
13/13 PASSED in 3.90s
├── TestCazellesIsolation: 7 tests
│   ├── construction (x=0.1, y=0.0, a=3.8)
│   ├── step() → int {0,1}
│   ├── spikes under drive
│   ├── slow variable y modulates x dynamics
│   ├── x clipped to [-2, 2]
│   ├── state finite (50k iterations)
│   └── reset()
├── TestCazellesNetwork: 3 tests
│   ├── Population(n=10)
│   ├── Network + PoissonInput → spikes
│   └── Projection(pop→pop) → spike_trains
└── TestCazellesAnalysis: 3 tests
    ├── firing_rate
    ├── spike_count
    └── isi (all > 0, all finite)
```

### Pipeline stages verified

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | ✓ PASS | x=0.1, y=0.0 |
| step() → int {0,1} | ✓ PASS | Level detection (x ≥ 0.9) |
| Spikes under drive | ✓ PASS | Fires with current |
| Slow y modulates | ✓ PASS | y changes → affects x dynamics |
| x clipped [−2, 2] | ✓ PASS | No divergence |
| State finite (50k) | ✓ PASS | x, y both finite |
| reset() | ✓ PASS | x→0.1, y→0.0 |
| Population(n=10) | ✓ PASS | 10 instances |
| Network + PoissonInput | ✓ PASS | Spikes produced |
| Projection(pop→pop) | ✓ PASS | spike_trains extractable |
| firing_rate | ✓ PASS | > 0 |
| spike_count | ✓ PASS | > 0 |
| isi | ✓ PASS | all > 0, all finite |

### Network configuration tested

- Population: 10 CazellesMapNeurons
- PoissonInput: rate=500Hz, weight sufficient for spiking
- Projection: self-recurrent
- SpikeMonitor: count, spike_trains, isi verified

**ALL 13 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**

---

## Numerical Considerations

- **No ODE:** Map iteration — no Euler error, no stiffness, no sub-stepping.
- **x clipped [−2, 2]:** Prevents divergence when y or I push x outside
  the logistic map's natural [0, 1] range.
- **y not clipped:** Slow variable can grow without bound. In practice,
  the ε(x − σ) dynamics self-limit y.
- **Pure arithmetic:** No exp, no cosh, no sqrt. Only multiply, add, clip.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/cazelles_map.py`.
- **Two state variables:** x (fast), y (slow).
- **Simplest map model** in SC-NeuroCore (tied with MedvedevMap).
- **Dataclass:** Uses `@dataclass`.
- **Polyglot `simulate`:** N-step recurrence accelerated by Rust (PyO3, engine
  `neurons/maps.rs`), Julia (`accel/julia/neurons/cazelles_map.jl`), Go (cgo,
  `accel/go/neurons/cazelles_map`) and Mojo (FFI, `accel/mojo/neurons`), all
  parity-checked against the NumPy reference. See *Polyglot acceleration* above.

---

## Performance

| Metric | Python | Notes |
|--------|--------|-------|
| Isolation | ~2M steps/s | No transcendental functions |
| Network (10n) | ~100K neuron-steps/s | Very fast |

Among the fastest models — pure arithmetic, no exp(), no ODE integration.

---

## Polyglot acceleration

The single `step` is trivial, but `simulate(n_steps, current, backend=...)` is a
sequential recurrence (each step depends on the previous) that does not
vectorise — a compiled inner loop genuinely beats Python. The kernel carries a
full polyglot chain:

```python
neuron = CazellesMapNeuron(a=3.8)
trace, spikes = neuron.simulate(2_000_000, current=0.05)            # auto -> Rust
trace, spikes = neuron.simulate(2_000_000, 0.05, backend="go")     # force a backend
```

`backend` accepts `"auto" | "rust" | "julia" | "go" | "mojo" | "python"`. `auto`
prefers Rust (it ships in the `sc_neurocore_engine` wheel) and falls back to the
pure-NumPy reference.

Because the map is exact floating-point arithmetic, **Rust, Julia and Go
reproduce the NumPy trace bit-for-bit**, even in the chaotic regime (a = 3.8).
Mojo's release build contracts `y + epsilon*(x - sigma)` into a fused
multiply-add (one rounding rather than two), so each step agrees to within two
ULP; in the chaotic regime the map amplifies that single ULP into a visible
trace gap, while the per-step physical-state agreement stays tightly bounded and
the spike counts still match. This is the documented Mojo FMA-parity behaviour,
not a defect.

### Measured backends

Reproduce with `python benchmarks/bench_cazelles_map.py --json
benchmarks/results/bench_cazelles_map.json`. Workload: 2,000,000 steps,
a = 3.8, median of 5 repeats. **Non-isolated** (loaded workstation, Python 3.12 /
NumPy 2.3) — functional/regression evidence, not isolated-core release numbers.

| backend | median (ms) | speedup vs NumPy | parity Δ vs NumPy |
|---|---:|---:|---:|
| python (NumPy) | 808.73 | 1.00× | 0 |
| go | 9.84 | 82.18× | 0 |
| mojo | 9.88 | 81.84× | 2.96e-04 (chaotic ULP amplification) |
| rust | 13.96 | 57.94× | 0 |
| julia | 25.75 | 31.40× | 0 |

Go and Mojo lead because they fill a preallocated NumPy buffer over the C ABI;
Rust returns a NumPy array directly (avoiding a multi-million-element Python-list
marshal); `auto` selects Rust as the always-available wheel backend within ~1.4×
of the fastest locally-built backends.

---

## Test Coverage Summary

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 7 | construction, binary, spikes, y modulates, x clipped, finite, reset |
| Network | 3 | Population, Network+spikes, Projection |
| Analysis | 3 | firing_rate, spike_count, isi |
| **Total** | **13** | **ALL PASSED (3.90s)** |

---

## Findings (Measured 2026-03-31)

1. **13/13 tests PASSED in 3.90s.** No failures.

2. **Slow y modulates fast x:** y changes under sustained x activity,
   confirming the two-timescale coupling.

3. **x clipped to [−2, 2]:** Prevents divergence while allowing
   subthreshold dynamics.

4. **Logistic map at a=3.8:** Near-chaotic regime — produces complex,
   non-periodic spike patterns within bursts.

5. **ε=0.01 creates 100:1 timescale separation:** Slow y drifts
   gradually while fast x iterates rapidly.

6. **No reset on spike:** x continues evolving — "spikes" are threshold
   crossings, not reset events.

7. **Network pipeline functional:** Population + PoissonInput + Projection
   all work. Map semantics differ from ODE models (no dt) but the
   pipeline handles this correctly.

8. **Computationally cheapest:** No transcendental functions, pure
   arithmetic — ~2M iterations/s.

9. **Deterministic:** No stochastic component. Same initial conditions
   → same trajectory (including chaotic sensitivity to perturbations).

10. **Map-based alternative to ODE bursters:** Achieves bursting without
    differential equations — computationally efficient for large networks.

---

## Theoretical Context

### Map-based neuron models in computational neuroscience

Map-based models (discrete-time) have a long history in theoretical
neuroscience:

- **Chialvo (1995):** Simplest 2D map producing excitable dynamics
- **Rulkov (2002):** Piecewise map with explicit spiking/bursting modes
- **Cazelles et al. (2001):** Logistic map + slow variable (this model)
- **Izhikevich (2003):** While technically an ODE, uses quadratic V
  nonlinearity that is map-like in discrete time
- **Ibarz & Tanaka (2004):** Piecewise-linear map with slow variable

### Why maps?

Maps are preferred when:
1. **Speed matters:** No Euler error, no dt stability issues, no exp()
2. **Qualitative dynamics matter more than waveform:** The exact shape
   of the action potential is unimportant
3. **Large network simulations:** 10,000+ neurons benefit from
   map efficiency
4. **Dynamical systems analysis:** Maps have a richer mathematical theory
   (Lyapunov exponents, symbolic dynamics, topological conjugacy)

### Logistic map and chaos theory

The logistic map f(x) = ax(1−x) is one of the foundational objects of
chaos theory:
- **Feigenbaum (1978):** Universal constants in period-doubling cascades
- **May (1976):** Logistic map as a model of population dynamics
- **Li & Yorke (1975):** "Period three implies chaos" — proved for maps

At a=3.8, the Cazelles model lives in the chaotic regime — the fast
dynamics are genuinely unpredictable in the long term, even though
the model is completely deterministic. This creates biologically
realistic spike timing variability from a purely deterministic mechanism.

The Cazelles model demonstrates that **chaos and bursting can coexist** —
the spike pattern within each burst is chaotic, while the burst envelope
is regular (controlled by the slow variable y).
