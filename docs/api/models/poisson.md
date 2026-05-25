# PoissonNeuron

**Module:** `sc_neurocore.neurons.models.poisson`
**Reference:** Standard Poisson process
**Family:** Statistical (rate-coded input generator)
**State variables:** None (stateless)

## Equations

$$P(\text{spike in } dt) = 1 - \exp\!\left(-\lambda \cdot dt / 1000\right)$$

No membrane dynamics. Pure Bernoulli trial each step.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rate_hz` | 100.0 | Firing rate (Hz) |
| `dt_ms` | 1.0 | Time step (ms) |

## Behaviour

- **Stateless:** No voltage, no memory. Each step is an independent Bernoulli trial.
- **Rate-coded:** Spike probability = 1 − exp(−λ·dt/1000). At rate=100Hz, dt=1ms: P≈0.09516.
- **rate_override:** `step(rate_override=X)` overrides stored rate. Negative value
  uses stored rate_hz (API convention).
- **No refractory period:** Consecutive spikes (ISI=1) are allowed.
- **ISI distribution:** Geometric (discrete analogue of exponential). Mean ISI = 1/p.
  CV(ISI) ≈ 1 for small p.
- **dt scaling:** Doubling dt doubles spike probability.
- **reset() is no-op:** Stateless — nothing to reset.

## Validation contract

The implementation revalidates mutable runtime parameters before every
Bernoulli sample:

- `rate_hz` must be finite and non-negative;
- `dt_ms` must be finite and positive;
- `rate_override` must be finite, with negative values selecting stored
  `rate_hz`;
- the interval hazard `rate_hz * dt_ms / 1000` must remain finite and
  non-negative before evaluating the finite-step probability;
- the resulting spike probability must remain finite and bounded in `[0, 1]`.

These guards preserve the Poisson-process interval law and prevent corrupted
mutable rate or timestep state from silently saturating to always-spike output.

## Statistical Properties

| Property | Value |
|----------|-------|
| Mean spike count | $N \cdot (1 - \exp(-\lambda \cdot dt / 1000))$ |
| Variance | $N \cdot p(1-p)$ |
| CV(ISI) | $\sqrt{1-p}/p \cdot p = \sqrt{1-p} \approx 1$ for small p |
| Mean ISI | $1 / (1 - \exp(-\lambda \cdot dt / 1000))$ steps |

## Infrastructure Pipeline

```
PoissonNeuron
├── step(rate_override?) → int {0,1} (stochastic)
├── Population: PoissonInput ignored (fires at own rate)
├── Verilog: LFSR comparator, ~15 LUTs
└── Rust/Go/Julia/Mojo: bounded interval probability contract
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 4 | defaults, binary, RNG init, reset no-op |
| Rate | 8 | mean matches finite-step Poisson probability, 4-point proportionality, monotonicity, zero rate, rate_override, negative override |
| ISI | 3 | geometric mean ISI, CV≈1, no refractory (ISI=1 exists) |
| dt scaling | 2 | dt doubles probability, small dt rare spikes |
| Validation | 18 | finite baseline rate, finite timestep, finite override, corrupted runtime state, finite interval hazard, bounded high-rate saturation |
| Stochasticity | 2 | different neurons differ, stateless (history-independent) |
| Network | 2 | population, spikes |
| Analysis | 2 | spike_count, consistency |
| **Total** | **42** | dedicated module checks |


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~402K steps/s |
| Spikes (10K steps, I=5.0) | 55 |
| State stability (20K steps) | PASS |
| Rust parity | N/A |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`PoissonNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
55 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(PoissonNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Polyglot mirrors
Rust and Go safety mirrors validate parameters and deterministically emit only
saturated-probability spikes with explicit errors for invalid state or
non-finite interval hazards. Julia uses the bounded probability with stochastic
Bernoulli sampling and raises `DomainError` for invalid contracts. Mojo exposes
the same finite-contract saturated-spike boundary with `-1` for invalid inputs.
Exact random-trace parity is not applicable to stochastic sampling.

---

## Findings (measured 2026-04-04)

1. Throughput: ~402K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Polyglot mirrors: bounded probability contract aligned with explicit invalid-contract signalling
4. Numerical stability confirmed over 20K steps
