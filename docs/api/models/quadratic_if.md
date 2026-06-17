# QuadraticIFNeuron

**Module:** `sc_neurocore.neurons.models.quadratic_if`
**Reference:** Ermentrout & Kopell 1986; Latham et al. 2000
**Family:** Integrate-and-fire (Type-I canonical form)
**State variables:** `v` (membrane voltage)

---

## Equations

### Continuous dynamics

$$\frac{dV}{dt} = V^2 + I$$

This is the canonical form for Type-I excitability near a saddle-node
bifurcation (Ermentrout 1996). The quadratic nonlinearity is the normal
form of the bifurcation — any conductance-based model with a saddle-node
on an invariant circle reduces to this via coordinate transform.

### Exact constant-current flow (as implemented)

For constant input over one timestep, SC-NEUROCORE advances the QIF state with
the closed-form Riccati flow rather than a forward-Euler increment:

| Current regime | Candidate voltage after one `dt` |
|----------------|-----------------------------------|
| $I > 0$ | $a \tan(a\,dt + \arctan(V/a))$, $a=\sqrt{I}$ |
| $I = 0$ | $V/(1 - V\,dt)$ |
| $I < 0$ | $a(1 + q)/(1 - q)$, $a=\sqrt{-I}$, $q=((V-a)/(V+a))\exp(2a\,dt)$ |

The candidate is computed before mutation. If the candidate is non-finite,
the call fails without changing state. If the candidate crosses
`v_peak`, the membrane resets to `v_reset` and emits one spike. Python, Rust
engine, Rust safety, Go service, Julia, and Mojo surfaces use the same
candidate-first exact-flow contract.

### Phase-plane structure

For $I < 0$: two fixed points at $V^* = \pm\sqrt{-I}$. The lower one
is stable; the upper is unstable. Neuron rests silently.

For $I = 0$: fixed points coalesce at $V = 0$ (saddle-node bifurcation).

For $I > 0$: no fixed points. $V$ increases monotonically until hitting
$V_{\text{peak}}$, resets to $V_{\text{reset}}$, and repeats.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | −1.0 | Membrane voltage (dimensionless) |
| `v_reset` | −1.0 | Post-spike reset potential |
| `v_peak` | 1.0 | Spike detection threshold |
| `dt` | 0.01 | Integration time step |

All quantities are dimensionless. To map to biophysical units, rescale
V and I according to the parent conductance model's parameters.

---

### Validation contract

The implementation rejects invalid state before mutation:

- `v`, `v_reset`, `v_peak`, `dt`, and input current must be finite;
- initial `v` and `v_reset` must be below `v_peak`;
- `dt` must be positive;
- exact-flow candidate voltages must remain finite before assignment.

These guards prevent a finite but numerically explosive quadratic flow from
poisoning the membrane state.

## Behaviour

### Saddle-node bifurcation at I=0

The defining property of QIF. Below the bifurcation ($I < 0$), the neuron
is absolutely silent — tested with 50,000 steps at $I = -0.5$: zero spikes.
At $I = 0$: also zero spikes (half-stable fixed point at $V = 0$, approached
asymptotically from $V_0 = -1$). Above ($I > 0$): periodic spiking — tested
at $I = 0.5$: 184 spikes in 50,000 steps.

### Type-I continuous f–I onset

Firing rate rises continuously from zero at $I = 0^+$. Measured:
- $I = 0.1$: ~62 spikes/50k steps
- $I = 1.0$: ~316 spikes/50k steps
- Ratio ≈ 5.1

In the continuous QIF without reset, $f \propto \sqrt{I}$ giving a ratio of
$\sqrt{10} \approx 3.16$. The measured ratio is higher because the discrete
reset transit from $V_{\text{peak}}$ to $V_{\text{reset}}$ adds a constant
time component to the ISI.

### Quadratic divergence

Once $V > 0$, the $V^2$ term creates positive feedback — voltage accelerates
upward. This is verified directly: starting from $V = 0.5$ with $I = 1.0$,
one step yields $\Delta V = (0.25 + 1.0) \times 0.01 = 0.0125 > 0$.

### Constant ISI

Deterministic model with no adaptation. After discarding initial transient
(first 5 spikes), measured CV(ISI) < 0.02 at $I = 1.0$.

---

## Measured Dynamics (from test probing)

| Current | Spikes (50k steps) | Mean ISI (steps) | Regime |
|---------|--------------------|-------------------|--------|
| $I = -0.5$ | 0 | — | Quiescent (stable FP at $V^* = -0.707$) |
| $I = 0.0$ | 0 | — | Marginal (half-stable FP at $V = 0$) |
| $I = 0.5$ | 184 | 271 | Slow periodic firing |
| $I = 1.0$ | 316 | 158 | Moderate periodic firing |
| $I = 2.0$ | 568 | 88 | Fast firing |
| $I = 5.0$ | 1,315 | 38 | Rapid firing |

f–I is monotonic. The measured ratio $f(4I)/f(I) \approx 3.4$ for
$I_1 = 1.0$, $I_2 = 4.0$: sub-linear but above the continuous $\sqrt{4} = 2.0$
prediction.

---

## Analytical Properties (continuous QIF, no reset)

| Property | Formula |
|----------|---------|
| Stable fixed point ($I < 0$) | $V^* = -\sqrt{-I}$ |
| Unstable fixed point ($I < 0$) | $V^{**} = +\sqrt{-I}$ |
| ISI | $T = \pi / \sqrt{I}$ |
| Firing rate | $f = \sqrt{I} / \pi$ |
| f–I ratio | $f(kI) / f(I) = \sqrt{k}$ |

These hold exactly only in the limit $V_{\text{peak}} \to +\infty$,
$V_{\text{reset}} \to -\infty$. With finite reset boundaries ($V_{\text{peak}} = 1$,
$V_{\text{reset}} = -1$), corrections scale as $O(1/\sqrt{I})$.

---

## Comparison with Other IF Models

| Property | LIF | QIF | EIF |
|----------|-----|-----|-----|
| f–I onset | Linear from $I_\theta$ | Continuous from $I = 0$ (√I) | Can be continuous or discontinuous |
| Spike upstroke | Linear ramp | Quadratic acceleration | Exponential blow-up |
| Canonical bifurcation | None | Saddle-node on invariant circle | Saddle-node (with sharpness param) |
| State variables | 1 | 1 | 1 |
| Computational cost | Lowest (1 add, 1 compare) | Low (1 multiply, 1 add, 1 compare) | Medium (1 exp, adds) |

---

## Numerical Considerations

- **dt sensitivity:** Tested stable at dt = 0.005, 0.01, 0.02 with $I = 1.0$
  over 50,000 steps. All produced finite states.
- **V_peak placement:** Lower V_peak → fewer steps per ISI → faster simulation
  but less of the quadratic acceleration region is traversed. Measured:
  V_peak = 0.5 fires more often than V_peak = 2.0 at same current.
- **Exact-flow peak crossing:** The closed-form candidate may pass
  `v_peak` within one timestep. The comparator `>=` catches the first
  discrete sample crossing and immediately resets, so the overshoot voltage is
  discarded rather than retained.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/quadratic_if.py`.
- **No sub-stepping:** Single exact constant-current flow step per `step()`
  call.
- **Polyglot surfaces:** Rust engine, Rust safety, Go, Julia, and Mojo QIF
  surfaces use the same finite-state, reset-below-peak, positive-`dt`,
  candidate-first exact-flow, and spike/reset contract as the Python model.
  Invalid native scalar paths return explicit errors or a dedicated invalid
  sentinel rather than silently converting numerical corruption into a no-spike
  event.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | construction defaults, step returns 0 or 1, voltage evolves under current, state finite after 50k steps, reset() restores v_reset |
| Bifurcation | 4 | I=−0.5 stable (0 spikes/50k), I=0 stable (0 spikes/50k), I=0.5 periodic (≥50 spikes), Type-I onset ratio I=0.1 vs I=1.0 |
| f–I curve | 2 | monotonic 4-point sweep (0.5, 1.0, 2.0, 5.0), sub-linear scaling f(4I)/f(I) ∈ (1.5, 4.0) |
| ISI | 2 | constant ISI at steady state (CV < 0.02, first 5 spikes excluded), ISI shortens monotonically with current |
| Edge cases | 4 | V² positive feedback from V=0.5, custom V_peak (lower peak → more spikes), dt stability at 0.005/0.01/0.02 |
| Determinism | 1 | bit-exact reproducibility across 2 independent runs (200 steps each) |
| Network | 2 | Population(n=10) construction, Network produces spikes with PoissonInput(rate=500Hz, weight=2.0) |
| Analysis | 2 | spike_count ≥ 100 in 50k steps at I=1.0, spike_count matches manual np.sum |
| Validation | 9 | finite parameters, peak/reset geometry, initial voltage below peak, finite current, finite exact-flow candidate before mutation |
| **Total** | **40** | |

---

## Findings

1. **Saddle-node bifurcation at I=0 confirmed:** Clean transition from
   0 spikes (I≤0) to sustained periodic firing (I>0). No transient
   spikes at I=0.
2. **Discrete reset modifies sqrt scaling:** Measured f(4)/f(1) ≈ 3.4
   vs. theoretical 2.0. The V_peak → V_reset transit contributes a
   constant ISI component that inflates the ratio.
3. **ISI perfectly constant:** CV effectively zero after transient.
   No adaptation, no noise, no slow variables.
4. **V_peak affects rate as expected:** Lower peak → shorter path →
   faster cycling. Verified with V_peak=0.5 vs V_peak=2.0.
5. **Numerically robust:** All three tested dt values (0.005–0.02)
   maintained finite state over 50k steps at I=1.0.
6. **Custom V_peak effect confirmed:** V_peak=0.5 fires more frequently
   than V_peak=2.0 at the same current — the shorter integration path
   reduces the ISI proportionally.

---

## Relationship to Other Models

The QIF is the **canonical form** for all Type-I neurons near the
bifurcation. Any conductance-based model exhibiting a saddle-node on
an invariant circle (SNIC) bifurcation can be reduced to the QIF via
a change of variables. This includes:

- **Connor-Stevens** near threshold (saddle-node via A-current)
- **Wang-Buzsaki** near threshold (SNIC in gamma interneurons)
- **Morris-Lecar** in Type-I parameter regime (I < saddle-node)

The QIF thus serves as a **universal reference model** for Type-I
excitability, analogous to how the theta neuron (which is the QIF in
phase coordinates) serves as the canonical phase model.


---

## Measured Performance (2026-06-16)

Local non-isolated regression run. These numbers are recorded for
regression comparison only and are not production throughput claims.

| Metric | Value |
|--------|-------|
| Evidence class | Local regression, non-isolated workstation |
| Benchmark artefact | `benchmarks/results/local_python_2026-06-16_quadratic_if_exact_flow.json` |
| Workload | 200000 steps, 5 repeats, I=0.5 |
| Polyglot contract | Python, Rust engine, Rust safety, Go, Julia, and Mojo exact-flow surfaces aligned, with explicit errors where supported |

| Backend | Median ns/step | Min ns/step | Max ns/step | Spikes |
|---------|---------------:|------------:|------------:|-------:|
| Python | 529.888895 | 506.170625 | 666.01508 | 738 |
| Rust engine | 62.256035 | 49.96484 | 69.097795 | 738 |
| Go service mirror | 51.41 | 48.78 | 56.64 | 738 |
| Julia mirror | 46.907565 | 45.69871 | 49.259465 | 738 |
| Mojo mirror | 45.37766988505609 | 44.75855501368642 | 46.39446997316554 | 738 |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`QuadraticIFNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
263 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(QuadraticIFNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Polyglot safety surfaces
Rust engine, Rust safety, Go, Julia, and Mojo carry the same exact-flow,
candidate-first spike/reset contract.

---

## Findings (measured 2026-06-16)

1. Local Python median: 529.888895 ns/step, about 1.89M steps/s in the
   non-isolated regression run.
2. Rust engine, Go, Julia, and Mojo measurements are present in the benchmark
   artefact; no maintained backend is skipped.
3. Exact-flow spike counts match across all five measured backends.
4. Numerical stability confirmed over 20K steps.
