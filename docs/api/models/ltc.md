# LiquidTimeConstantNeuron

**Module:** `sc_neurocore.neurons.models.ltc`
**Reference:** Hasani et al. 2021 (NeurIPS)
**Family:** Integrate-and-fire (input-adaptive time constant)
**State variables:** `x` (hidden state)

## Equations

$$\tau(x,I) = \tau_{base} \cdot \sigma(w_\tau \cdot I + b)$$
$$f(x,I) = \tanh(w_x \cdot x + w_{in} \cdot I)$$
$$x(t+1) = x(t) + \frac{dt}{\tau} \cdot (-x + f)$$

Spike: $x \geq V_\theta$, hard reset $x \to 0$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_base` | 10.0 | Base time constant |
| `w_tau` | -0.5 | Input → tau coupling |
| `w_x` | 0.8 | Self-coupling weight |
| `w_in` | 1.0 | Input weight |
| `v_threshold` | 1.0 | Spike threshold |

## Behaviour

- **Input-adaptive tau:** Time constant changes with input via sigmoid.
  Larger input → faster dynamics (w_tau < 0).
- **Sharp transition:** I ∈ [4, 4.5] is the critical range. Below:
  x settles to ~0.999 (subthreshold). Above: spikes every step.
- **tanh saturation:** f_target saturates at ±1, so x cannot exceed 1.0
  unless driven past the tanh ceiling.
- **NeurIPS 2021:** Designed for continuous-time sequence modelling.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, silent, subthreshold settle, spikes, sharp transition, tau input-dependent, tanh target, stability, reset, deterministic |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **14** | |


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~157K steps/s |
| Spikes (10K steps, I=5.0) | 10000 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`LiquidTimeConstantNeuron()` instantiates with documented defaults.
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
`Population(LiquidTimeConstantNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~157K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps
