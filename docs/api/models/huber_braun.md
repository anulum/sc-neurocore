# HuberBraunNeuron

**Module:** `sc_neurocore.neurons.models.huber_braun`
**Reference:** Braun, Huber et al. 1998
**Family:** Conductance-based (cold receptor, temperature-dependent)
**State variables:** `v`, `a_sd` (slow depolarising), `a_sr` (slow repolarising)

## Equations

$$\frac{dV}{dt} = -g_{sd} a_{sd}(V-E_{sd}) - g_{sr} a_{sr}(V-E_{sr}) - g_L(V-E_L) + I + \eta\xi(t)$$
$$\tau_{sd} \frac{da_{sd}}{dt} = \sigma_{sd}(V) - a_{sd}$$
$$\tau_{sr} \frac{da_{sr}}{dt} = \sigma_{sr}(V) - a_{sr}$$

where $\sigma$ are sigmoid activation functions and $\xi(t)$ is Gaussian noise.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_sd` | 1.5 | Slow depolarising conductance |
| `g_sr` | 0.4 | Slow repolarising conductance |
| `g_l` | 0.1 | Leak conductance |
| `tau_sd` | 10.0 | SD time constant (ms) |
| `tau_sr` | 20.0 | SR time constant (ms) |
| `eta` | 0.012 | Noise amplitude |
| `dt` | 0.1 | Integration step (ms) |

## Behaviour

- **Cold receptor:** Models temperature-sensitive neurons in the skin.
  Oscillation regime depends on temperature (mapped to conductance ratios).
- **Default params:** Produce a single spike then settle to depolarised
  equilibrium (~+46 mV). Sustained oscillation requires parameter tuning.
- **Stochastic:** Gaussian noise (eta > 0) can trigger stochastic resonance.
- **No fast Na inactivation:** Simplified model — lacks repolarisation
  mechanism for sustained spiking in default regime.

## Infrastructure Pipeline

```
HuberBraunNeuron
├── step(current) → int {0,1} (threshold crossing)
├── Population: works
├── Verilog: 2 sigmoid LUTs + noise LFSR, ~80 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, initial spike, sd gating, sr gating, noise present, no noise deterministic, stability, gating bounded, reset, depolarised equilibrium |
| Network | 1 | Population |
| Analysis | 1 | spike_count |
| **Total** | **13** | |


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~73K steps/s |
| Spikes (10K steps, I=5.0) | 1 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`HuberBraunNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
1 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(HuberBraunNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~73K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps
