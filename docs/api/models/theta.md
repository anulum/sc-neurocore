# ThetaNeuron

**Module:** `sc_neurocore.neurons.models.theta`
**Reference:** Ermentrout & Kopell 1986
**Family:** Phase model (canonical Type-I on unit circle)
**State variables:** `theta` (phase angle, wrapped to [−π, π])

---

## Equations

### Phase dynamics

$$\frac{d\theta}{dt} = (1 - \cos\theta) + (1 + \cos\theta) \cdot I$$

### Spike detection

Upward crossing of $0.99\pi$ (slightly below π to avoid numerical issues):

$$\text{spike} = \begin{cases} 1 & \text{if } \theta_{prev} < 0.99\pi \text{ and } \theta \geq 0.99\pi \\ 0 & \text{otherwise} \end{cases}$$

### Phase wrapping

After each step: $\theta \leftarrow ((\theta + \pi) \bmod 2\pi) - \pi$.

### Relationship to QIF

The theta neuron is the **QIF mapped to the unit circle** via:

$$V = \tan(\theta/2)$$

This transforms the infinite V range of QIF into the bounded $\theta \in [-\pi, \pi]$.
The dynamics, bifurcation structure, and f–I curve are mathematically identical.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `theta` | 0.0 | Phase angle (radians) |
| `dt` | 0.01 | Time step |

---

### Validation contract

The implementation preserves the compact-circle state contract before mutation:

- initial `theta`, `dt`, and input current must be finite;
- `dt` must be positive;
- initial `theta` is normalised into `[-pi, pi]`;
- each tangent-half-angle exact-flow candidate phase must remain finite before assignment.
- runtime `theta` and `dt` are revalidated before exact-flow evaluation so
  corrupted objects fail closed without mutating phase.

These guards prevent finite but numerically explosive inputs from turning the
phase state into `NaN` while preserving the theta/QIF phase-map semantics.
Native Go and Rust mirrors return explicit errors for invalid scalar state,
Julia throws `DomainError`, and Mojo returns `-1` as the invalid sentinel.

## Behaviour

### Saddle-node bifurcation at I=0

At I<0: two fixed points on the circle (one stable, one unstable). The
neuron rests at the stable point $\theta^* = -\arccos\!\bigl(\frac{1+I}{1-I}\bigr)$.

At I=0: fixed points coalesce. theta remains at 0.

At I>0: no fixed points. theta increases monotonically (with angular
velocity depending on position), cycling through the full circle. Each
crossing of ~π constitutes a spike.

### Analytical ISI = π/√I

In continuous time, the period of the limit cycle is:

$$T = \frac{\pi}{\sqrt{I}}$$

This gives the firing rate $f = \sqrt{I}/\pi$ Hz. Verified at four
current levels — all match to within 2%:

| Current | ISI (measured, steps) | ISI × dt (time) | π/√I (analytical) | Error |
|---------|----------------------|------------------|---------------------|-------|
| 0.5 | 444 | 4.44 | 4.443 | < 0.1% |
| 1.0 | 314 | 3.14 | 3.142 | < 0.1% |
| 2.0 | 222 | 2.22 | 2.221 | < 0.1% |
| 5.0 | 140 | 1.40 | 1.405 | < 0.4% |

### √I rate scaling

$f(4I)/f(I) = \sqrt{4} = 2.0$. Measured: ratio between I=4 and I=1
rates is within (1.8, 2.2).

### Fixed point at negative I

At I=−0.5: $\theta^* = -\arccos(1/3) \approx -1.231$. Measured:
theta converges to within 0.01 of this value after 100k steps.

### Near-constant ISI

ISI alternates between floor and ceil of the analytical value (e.g.,
314 and 315 at I=1.0) due to discrete spike detection at 0.99π.
Only 2 unique ISI values, differing by exactly 1 step.

---

## Measured Dynamics

| Current | Spikes (50k) | Mean ISI | Regime |
|---------|-------------|----------|--------|
| −1.0 | 0 | — | Stable FP at θ = −π/2 |
| −0.5 | 0 | — | Stable FP at θ ≈ −1.231 |
| 0.0 | 0 | — | FP at θ = 0 |
| 0.1 | 50 | 993 | Very slow cycling |
| 0.5 | 113 | 444 | Slow cycling |
| 1.0 | 159 | 314 | Moderate cycling |
| 2.0 | 225 | 222 | Fast cycling |
| 5.0 | 356 | 140 | Rapid cycling |

---

## Comparison with QIF

| Property | QIF | Theta |
|----------|-----|-------|
| State variable | V ∈ (−∞, ∞) | θ ∈ [−π, π] |
| Spike | V ≥ V_peak → reset | θ crosses ~π → wrap |
| Fixed points (I<0) | V* = ±√(−I) | θ* = ±arccos((1+I)/(1-I)) |
| ISI (continuous) | π/√I | π/√I (identical) |
| f–I scaling | √I / π | √I / π (identical) |
| Numerical range | V can diverge | θ always bounded |
| Reset mechanism | Hard reset V → V_reset | Phase wrapping (continuous) |

The theta neuron is preferred for analytical work because the state is
bounded (no divergence risk) and the phase space is compact (S¹).

---

## Numerical Considerations

- **Phase wrapping ensures boundedness:** theta never diverges because it's
  wrapped to [−π, π] at construction and after every accepted step. Non-finite
  exact-flow candidates are rejected before state mutation.
- **0.99π detection threshold:** The spike detection uses 0.99π instead of
  exact π to avoid missing spikes due to discrete stepping. This introduces
  ±1 step ISI jitter.
- **dt invariance of ISI_time:** Verified: ISI_steps × dt gives the same
  physical time at dt=0.01 and dt=0.005 (within 0.1 time units).
- **cos() calls:** Two per step. These are the dominant computational cost.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/theta.py` — 36 lines.
- **NumPy dependency:** `np.cos` and `np.pi` for phase dynamics.
- **Polyglot surfaces:** Rust, Go, Julia, and Mojo theta surfaces use the same finite-state, compact-phase, positive-`dt`, finite-increment, and spike-crossing contract as the Python model.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 6 | defaults, binary, evolution, wrapping [−π,π], finite 100k, reset |
| Bifurcation | 5 | I<0 silent (2 values), I=0 stays at 0, I>0 fires, continuous onset, fixed point θ* verified analytically |
| Analytical ISI | 6 | ISI×dt matches π/√I to <2% (4 currents), near-constant ISI (±1 step), √I scaling f(4I)/f(I)≈2 |
| Phase space | 3 | full circle traversal, spike at ~π, dynamics equation dθ/dt check |
| Parameters | 4 | dt stability (3 values), dt invariance of ISI_time |
| Edge cases | 2 | wrapping under large dt, deterministic |
| **Pipeline** | 4 | **Population, Network+PoissonInput, Projection src→tgt propagation, full analysis (spike_count + isi + firing_rate cross-validated)** |
| Validation | 10 | finite phase/current/dt, compact initial phase, finite phase increment before mutation |
| **Total** | **42** | |

---

## Findings

1. **ISI = π/√I verified to < 0.4%** at all four tested currents. The
   discrete-time simulation matches the continuous analytical prediction
   with remarkable precision.
2. **Fixed point at I=−0.5 verified:** theta converges to −arccos(1/3) ≈
   −1.231 within 0.01 after 100k steps.
3. **√I rate scaling confirmed:** f(4)/f(1) = 2.0 ± 0.2.
4. **Phase wrapping ensures numerical robustness:** theta stays in [−π, π]
   regardless of input magnitude or simulation length.
5. **ISI jitter = ±1 step:** The 0.99π detection threshold causes ISI to
   alternate between floor and ceil. Only 2 unique ISI values observed.
6. **dt invariance:** ISI in physical time units is the same at dt=0.01
   and dt=0.005 (within 0.1 time units).
7. **Projection wiring confirmed:** Source neurons drive target neurons
   through Projection. Both source and target produce spikes.


---

## Measured Performance (2026-06-16)

Local non-isolated regression run. These numbers are recorded for
regression comparison only and are not production throughput claims.

| Metric | Value |
|--------|-------|
| Evidence class | Local regression, non-isolated workstation |
| Benchmark artefact | `benchmarks/results/local_python_2026-06-16_theta_exact_flow.json` |
| Workload | 200000 steps, 5 repeats, I=0.5 |
| Polyglot contract | Python, Rust engine, Go, Julia, and Mojo spike-kernel surfaces aligned where maintained |

| Backend | Median ns/step | Min ns/step | Max ns/step | Spikes |
|---------|---------------:|------------:|------------:|-------:|
| Python | 1115.22545 | 1019.392715 | 1574.31564 | 450 |
| Rust engine | 138.5801 | 120.986305 | 147.37324 | 450 |
| Go service mirror | 156.5 | 150.7 | 179.7 | 450 |
| Julia mirror | 127.33956 | 126.497845 | 128.39905 | 450 |
| Mojo mirror | 104.11917013698258 | 98.57875003945082 | 106.32325502228923 | 450 |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`ThetaNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
71 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(ThetaNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Polyglot safety surfaces
Rust, Go, Julia, and Mojo carry the same spike-crossing and compact-phase validation contract.

---

## Findings (measured 2026-06-16)

1. Local Python median: 1115.22545 ns/step, about 897K steps/s in the
   non-isolated regression run.
2. Rust engine, Go, Julia, and Mojo measurements are present in the benchmark
   artefact; no maintained backend is skipped.
3. Polyglot contract aligned for Rust, Go, Julia, and Mojo.
4. Numerical stability confirmed over 20K steps.
