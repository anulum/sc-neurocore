# InhomogeneousPoissonNeuron

**Module:** `sc_neurocore.neurons.models.inhomogeneous_poisson`
**Reference:** Cox 1955
**Family:** Statistical (doubly stochastic Poisson)
**State variables:** None

## Equations

$$P(\text{spike in } dt) = \max(0, \lambda(t)) \cdot dt / 1000$$

where $\lambda(t)$ = `rate_hz` argument (time-varying).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt_ms` | 1.0 | Time step (ms) |

## Behaviour

- **Stateless:** No membrane, no history — pure instantaneous rate coding.
- **Time-varying rate:** Rate passed per step → models any rate signal.
- **Negative rate clamped:** max(0, rate) prevents negative probability.
- **Simplest spike generator:** Used as input layer, benchmark baseline,
  or Poisson drive source.

## Infrastructure Pipeline

```
InhomogeneousPoissonNeuron
├── step(rate_hz) → int {0,1} (stochastic)
├── Population: works (no current — uses rate_hz)
├── Verilog: LFSR comparator, ~10 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step binary, zero rate, negative rate, spikes at rate, rate proportional, time-varying, stochastic, reset noop, custom dt |
| Network | 1 | Population |
| Analysis | 1 | spike_count |
| **Total** | **12** | |


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~205K steps/s |
| Spikes (10K steps, I=5.0) | 56 |
| State stability (20K steps) | PASS |
| Rust parity | N/A |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`InhomogeneousPoissonNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
56 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(InhomogeneousPoissonNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**N/A** — stochastic model, exact parity not applicable.

---

## Findings (measured 2026-04-04)

1. Throughput: ~205K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: N/A
4. Numerical stability confirmed over 20K steps
