# GalvesLocherbachNeuron

**Module:** `sc_neurocore.neurons.models.galves_locherbach`
**Reference:** Galves & Löcherbach 2013
**Family:** Stochastic (point process)
**State variables:** `v` (membrane potential — accumulator, not ODE)

## Equations

$$V(t) = \gamma \cdot V(t-1) + w_{\text{input}}$$
$$P(\text{spike}) = \sigma\bigl(s \cdot (V - V_\theta)\bigr) \cdot dt$$

where $\sigma$ is the logistic sigmoid, $\gamma$ is decay, $s$ is steepness.

No ODE — purely probabilistic spiking with leaky integration.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `decay` | 0.95 | Voltage decay factor per step |
| `threshold_rate` | 0.5 | Centre of sigmoid (half-max firing) |
| `steepness` | 5.0 | Sigmoid sharpness |
| `dt` | 1.0 | Time step |

## Behaviour

- **Stochastic:** No deterministic threshold — spike probability is sigmoid
  of voltage. High steepness ≈ hard threshold.
- **Leaky accumulator:** Voltage decays by factor `decay` each step,
  accumulates weighted input.
- **Reset on spike:** v → v_rest after each spike.
- **Point process:** Mathematically rigorous stochastic neural model
  from probability theory (not biophysics).

## Infrastructure Pipeline

```
GalvesLocherbachNeuron
├── step(weighted_input) → int {0,1} (stochastic)
├── Population: PoissonInput(weight=1.0, rate=500Hz)
├── Verilog: sigmoid LUT + LFSR, ~40 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, stochastic spiking, rate increase, sigmoid probability, decay, reset on spike, stability, reset, custom steepness, low drive |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **14** | |


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~38K steps/s |
| Spikes (10K steps, I=5.0) | 10000 |
| State stability (20K steps) | PASS |
| Rust parity | N/A |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`GalvesLocherbachNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
10000 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(GalvesLocherbachNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**N/A** — stochastic model, exact parity not applicable.

---

## Findings (measured 2026-04-04)

1. Throughput: ~38K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: N/A
4. Numerical stability confirmed over 20K steps
