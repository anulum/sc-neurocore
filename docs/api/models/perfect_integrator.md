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

## Validation contract

The implementation rejects invalid state before mutation:

- `v`, `v_threshold`, `v_reset`, `c_m`, `dt`, and input current must be finite;
- `c_m` and `dt` must be positive;
- `v_threshold` must be greater than `v_reset`;
- initial `v` must be below `v_threshold`;
- each voltage increment and candidate voltage must remain finite before assignment.
- runtime `v`, `c_m`, `dt`, `v_threshold`, and `v_reset` are revalidated before
  the `I / C_m` division so corrupted objects fail closed without mutating
  voltage.

These guards preserve the analytical positive-excursion ISI contract and prevent
overflowing currents or capacitance scales from poisoning the state.
Native Go and Rust mirrors return explicit errors for invalid scalar state,
Julia throws `DomainError`, and Mojo returns `-1` as the invalid scalar sentinel.

The schema-level reference-trace corpus also pins a spike-bearing constant-current
protocol, `perfect_integrator_constant_current_sawtooth`. That trace records
post-reset states from the analytic sawtooth solution and validates spike count,
first-spike step, and final/min/max/mean voltage through the public
`UniversalNeuron` runner.

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
└── Rust/Go/Julia/Mojo: same finite-increment spike/reset contract
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary return, zero-drift, linear ramp, no-leak invariant |
| Threshold | 4 | exact threshold, reset, custom reset, superthreshold |
| Analytical f–I | 7 | 5-point f–I curve, linearity ratio, threshold/capacitance dependence |
| ISI analysis | 3 | constant ISI, analytical match, CV=0 |
| Edge cases | 10 | negative current, large negative, small/large dt, threshold-reset rejection, FP accumulation, alternating I, reset(), determinism |
| Parameter sweep | 8 | 4 C_m values, 4 threshold values (rate ∝ 1/param) |
| Network | 3 | population, spikes, two-population drive comparison |
| Analysis | 2 | spike_count manual match, long-run analytical match |
| Validation | 19 | finite parameters/current, positive scales, positive threshold excursion, initial voltage below threshold, finite increment before mutation |
| **Total** | **60** | |

Finding: floating-point accumulation of 0.1 can delay spike by 1 step when
the analytical ISI is exactly N steps. Documented in test and behaviour section.


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~572K steps/s |
| Spikes (10K steps, I=5.0) | 5000 |
| State stability (20K steps) | PASS |
| Polyglot contract | Rust, Go, Julia, and Mojo finite-increment surfaces aligned |

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

### 7. Polyglot safety surfaces
Rust, Go, Julia, and Mojo carry the same finite-increment spike/reset contract.

---

## Findings (measured 2026-04-04)

1. Throughput: ~572K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Polyglot contract aligned for Rust, Go, Julia, and Mojo
4. Numerical stability confirmed over 20K steps
