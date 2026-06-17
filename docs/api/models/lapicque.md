# LapicqueNeuron

**Module:** `sc_neurocore.neurons.models.lapicque`
**Reference:** Lapicque 1907
**Family:** Integrate-and-fire (classical)
**State variables:** `v` (voltage)

## Equations

$$\tau \frac{dV}{dt} = -(V - V_r) + R \cdot I$$

Spike: $V \geq V_\theta$, hard reset $V \to V_{reset}$.

For constant current over a timestep, maintained runtime surfaces use the exact
RC flow rather than forward Euler:

$$V_{t+\Delta t} = V_\infty + (V_t - V_\infty)e^{-\Delta t/\tau}$$

where $V_\infty = V_r + R \cdot I$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 20.0 | Membrane time constant (ms) |
| `resistance` | 1.0 | Membrane resistance |
| `v_threshold` | 1.0 | Spike threshold |
| `v_reset` | 0.0 | Post-spike reset |
| `dt` | 1.0 | Integration step |

## Validation contract

The implementation rejects invalid state before mutation:

- `v`, `v_rest`, `v_reset`, `v_threshold`, `tau`, `resistance`, `dt`, and input current must be finite;
- `tau`, `resistance`, and `dt` must be positive;
- `v_threshold` must be greater than both `v_rest` and `v_reset`;
- initial `v` must be below `v_threshold`;
- exact-flow steady voltage, decay, and candidate voltage must remain finite
  before assignment.

These guards preserve the positive-rheobase RC contract and prevent overflowing
inputs or time constants from poisoning membrane state.

Python re-validates mutable runtime state on every `step()` call. Rust and Go
return explicit errors for invalid currents, corrupted state, or non-finite
exact-flow candidates; Julia raises `DomainError` for the same contract. The Mojo kernel
surface remains a pure spike-flag function and fails closed with `0` for invalid
inputs.

## Behaviour

- **The original IF:** Lapicque 1907 — the first mathematical neuron model.
  Simple RC circuit with threshold.
- **Analytical rheobase:** I_rh = V_θ / R. Below rheobase, v settles to
  steady state R·I < V_θ. Above, periodic spiking.
- **Deterministic:** Fully deterministic exact constant-current RC integration.
- **Hard reset:** v → v_reset (not subtract-reset).
- **Simplest conductance-free model:** No gating, no adaptation, no noise.

## Infrastructure Pipeline

```
LapicqueNeuron
├── step(current) → int {0,1}
├── Population: PoissonInput(weight=2.0, rate=500Hz)
├── Verilog: MAC + compare, ~10 LUTs
└── Rust/Go/Julia/Mojo: finite exact-flow spike/reset contract with explicit errors where supported
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 12 | construction, step binary, subthreshold, spikes, rheobase, rate increase, voltage clamp, hard reset, stability, reset, deterministic, custom tau |
| Network | 2 | Population, spikes |
| Analysis | 4 | spike_count, ISI, firing-rate, cross-validation |
| Validation | 27 | finite parameters/current, positive RC scales, threshold geometry, corrupted runtime state, initial voltage below threshold, finite candidate before mutation |
| Exact flow | 2 | closed-form update and separation from forward Euler |
| **Total** | **68** | |


---

## Measured Performance (2026-06-17)

Local non-isolated regression run. These numbers are recorded for
regression comparison only and are not production throughput claims.

| Metric | Value |
|--------|-------|
| Evidence class | Local regression, non-isolated workstation |
| Benchmark artefact | `benchmarks/results/local_python_2026-06-17_lapicque_exact_flow.json` |
| Workload | 200000 steps, 5 repeats, I=5.0 |
| Polyglot contract | Python, Rust engine, Rust safety, Go, Julia, and Mojo exact-flow surfaces aligned, with explicit errors where supported |

| Backend | Median ns/step | Min ns/step | Max ns/step | Spikes |
|---------|---------------:|------------:|------------:|-------:|
| Python | 1078.79079 | 1019.581435 | 1252.41781 | 40000 |
| Rust engine | 42.9045 | 40.04455 | 93.734405 | 40000 |
| Go service mirror | 47.22 | 37.41 | 54.99 | 40000 |
| Julia mirror | 13.54947 | 12.98847 | 15.83982 | 40000 |
| Mojo mirror | 3.993540012743324 | 3.8065949047449976 | 4.111330054001883 | 40000 |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`LapicqueNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
2000 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(LapicqueNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Polyglot safety surfaces
Rust, Go, Julia, and Mojo carry the same finite exact-flow spike/reset contract.

---

## Findings (measured 2026-06-17)

1. Constant-current integration now uses the closed-form RC update.
2. All pipeline stages verified green.
3. Polyglot contract aligned for Python, Rust engine, Rust safety, Go, Julia, and Mojo.
4. Numerical stability confirmed over 20K steps.
