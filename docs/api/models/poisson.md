# PoissonNeuron

**Module:** `sc_neurocore.neurons.models.poisson`
**Reference:** Standard Poisson process
**Family:** Statistical (rate-coded input generator)
**State variables:** None (stateless)

## Equations

$$P(\text{spike in } dt) = \lambda \cdot dt / 1000$$

No membrane dynamics. Pure Bernoulli trial each step.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rate_hz` | 100.0 | Firing rate (Hz) |
| `dt_ms` | 1.0 | Time step (ms) |

## Behaviour

- **Stateless:** No voltage, no memory. Each step is an independent Bernoulli trial.
- **Rate-coded:** Spike probability = λ·dt/1000. At rate=100Hz, dt=1ms: P=0.1.
- **rate_override:** `step(rate_override=X)` overrides stored rate. Negative value
  uses stored rate_hz (API convention).
- **No refractory period:** Consecutive spikes (ISI=1) are allowed.
- **ISI distribution:** Geometric (discrete analogue of exponential). Mean ISI = 1/p.
  CV(ISI) ≈ 1 for small p.
- **dt scaling:** Doubling dt doubles spike probability.
- **reset() is no-op:** Stateless — nothing to reset.

## Statistical Properties

| Property | Value |
|----------|-------|
| Mean spike count | $N \cdot \lambda \cdot dt / 1000$ |
| Variance | $N \cdot p(1-p)$ |
| CV(ISI) | $\sqrt{1-p}/p \cdot p = \sqrt{1-p} \approx 1$ for small p |
| Mean ISI | $1000 / (\lambda \cdot dt)$ steps |

## Infrastructure Pipeline

```
PoissonNeuron
├── step(rate_override?) → int {0,1} (stochastic)
├── Population: PoissonInput ignored (fires at own rate)
├── Verilog: LFSR comparator, ~15 LUTs
└── Rust: supported (stateless)
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 4 | defaults, binary, RNG init, reset no-op |
| Rate | 8 | mean matches λ, 4-point proportionality, monotonicity, zero rate, rate_override, negative override |
| ISI | 3 | geometric mean ISI, CV≈1, no refractory (ISI=1 exists) |
| dt scaling | 2 | dt doubles probability, small dt rare spikes |
| Stochasticity | 2 | different neurons differ, stateless (history-independent) |
| Network | 2 | population, spikes |
| Analysis | 2 | spike_count, consistency |
| **Total** | **24** | |


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

### 7. Rust parity
**N/A** — stochastic model, exact parity not applicable.

---

## Findings (measured 2026-04-04)

1. Throughput: ~402K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: N/A
4. Numerical stability confirmed over 20K steps
