# LearnableNeuronModel

**Module:** `sc_neurocore.neurons.models.lnm`
**Reference:** Jahns et al. 2025
**Family:** Integrate-and-fire (fully learnable)
**State variables:** `v` (voltage)

## Equations

$$v(t) = \alpha \cdot v(t-1) + \beta \cdot I(t) + \gamma \cdot \sigma(v(t-1))$$

where $\sigma(v) = 1/(1+\exp(-s(v-c)))$.

Spike: $v \geq V_\theta$, hard reset $v \to 0$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.9 | Voltage decay (trainable) |
| `beta` | 0.1 | Input scaling (trainable) |
| `gamma` | 0.05 | Nonlinear feedback (trainable) |
| `f_slope` | 5.0 | Sigmoid steepness |
| `f_shift` | 0.5 | Sigmoid centre |
| `v_threshold` | 1.0 | Spike threshold |

## Behaviour

- **Fully trainable:** All 3 core params (alpha, beta, gamma) are
  differentiable — designed for gradient-based SNN optimisation.
- **Nonlinear feedback:** gamma * sigmoid(v) adds voltage-dependent
  self-excitation. gamma=0 reduces to linear LIF.
- **Hard reset:** v → 0 on spike.
- **Deterministic:** No stochastic element.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, silent, spikes, rate increase, alpha effect, beta effect, gamma=0 linear, stability, reset, deterministic |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **14** | |


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~327K steps/s |
| Spikes (10K steps, I=5.0) | 3333 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`LearnableNeuronModel()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
3333 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(LearnableNeuronModel, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~327K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps
