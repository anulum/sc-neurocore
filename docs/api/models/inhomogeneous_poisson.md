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
