# PerfectIntegratorNeuron

**Module:** `sc_neurocore.neurons.models.perfect_integrator`
**Reference:** Lapicque 1907 (no-leak variant)
**Family:** Integrate-and-fire (non-leaky)
**State variables:** `v` (voltage)

## Equations

$$C_m \frac{dV}{dt} = I$$

Discrete: $V(t+1) = V(t) + \frac{I}{C_m} \cdot dt$

Spike when $V \geq V_\theta$, then $V \leftarrow V_{\text{reset}}$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | 0.0 | Membrane voltage |
| `c_m` | 1.0 | Membrane capacitance |
| `v_threshold` | 1.0 | Spike threshold |
| `v_reset` | 0.0 | Reset potential |
| `dt` | 0.1 | Time step |

## Behaviour

- **No leak:** Voltage accumulates indefinitely — zero-input steps leave V unchanged.
  This is the key distinction from LIF (where V decays toward rest).
- **Perfectly linear f–I curve:** Firing rate $f = \frac{I \cdot dt}{C_m \cdot (\theta - V_{\text{reset}})}$,
  clamped at 1 spike/step maximum.
- **Deterministic:** Identical inputs produce bit-for-bit identical spike trains.
- **Constant ISI:** All inter-spike intervals are identical for constant input
  (CV(ISI) = 0 exactly).
- **Floating-point caveat:** With $I=1, C_m=1, dt=0.1$: ten additions of 0.1
  yield $V \approx 0.9999\ldots$ due to IEEE 754, delaying the spike by one step.

## Analytical Predictions

| Property | Formula |
|----------|---------|
| ISI (steps) | $\lceil (\theta - V_{\text{reset}}) / (I \cdot dt / C_m) \rceil$ |
| Rate (Hz) | $I / (C_m \cdot (\theta - V_{\text{reset}}))$ (continuous) |
| Linearity | $f(2I) = 2 f(I)$ exactly |
| Capacitance scaling | $f \propto 1/C_m$ |
| Threshold scaling | $f \propto 1/\theta$ |

## Infrastructure Pipeline

```
PerfectIntegratorNeuron
├── step(current) → int {0,1} (deterministic)
├── Population: PoissonInput(weight=5, rate=500Hz)
├── Verilog: accumulator + comparator, ~20 LUTs
└── Rust: supported (single f64 state)
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary return, zero-drift, linear ramp, no-leak invariant |
| Threshold | 4 | exact threshold, reset, custom reset, superthreshold |
| Analytical f–I | 7 | 5-point f–I curve, linearity ratio, threshold/capacitance dependence |
| ISI analysis | 3 | constant ISI, analytical match, CV=0 |
| Edge cases | 10 | negative current, large negative, small/large dt, θ=reset, FP accumulation, alternating I, reset(), determinism |
| Parameter sweep | 8 | 4 C_m values, 4 threshold values (rate ∝ 1/param) |
| Network | 3 | population, spikes, two-population drive comparison |
| Analysis | 2 | spike_count manual match, long-run analytical match |
| **Total** | **42** | |

Finding: floating-point accumulation of 0.1 can delay spike by 1 step when
the analytical ISI is exactly N steps. Documented in test and behaviour section.


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~572K steps/s |
| Spikes (10K steps, I=5.0) | 5000 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`PerfectIntegratorNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
5000 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(PerfectIntegratorNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~572K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps
