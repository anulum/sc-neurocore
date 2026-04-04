# GatedLIFNeuron

**Module:** `sc_neurocore.neurons.models.gated_lif`
**Reference:** Yao et al. 2022 (NeurIPS)
**Family:** Integrate-and-fire (learnable gated)
**State variables:** `v` (voltage)

## Equations

$$v(t) = g_v \cdot v(t-1) + g_i \cdot I(t)$$

Spike: $v \geq V_\theta$, subtract-reset: $v \leftarrow v - V_\theta$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gate_v` | 0.9 | Voltage decay gate (learnable in training) |
| `gate_i` | 1.0 | Input scaling gate (learnable in training) |
| `v_threshold` | 1.0 | Spike threshold |
| `dt` | 1.0 | Time step |

## Behaviour

- **Learnable gates:** `gate_v` and `gate_i` are trainable parameters
  in SNN training frameworks. Replaces fixed decay constant.
- **Subtract-reset:** v -= V_θ on spike (preserves excess voltage).
- **Deterministic:** No stochastic element — identical input = identical output.
- **AI-optimised:** Designed for deep SNN training, not biophysics.

## Infrastructure Pipeline

```
GatedLIFNeuron
├── step(current) → int {0,1}
├── Population: PoissonInput(weight=0.5, rate=500Hz)
├── Verilog: 2 multiplies + compare, ~15 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, subthreshold, spikes, subtract reset, rate increase, gate_v effect, gate_i effect, stability, reset, deterministic |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **14** | |


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~434K steps/s |
| Spikes (10K steps, I=5.0) | 10000 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`GatedLIFNeuron()` instantiates with documented defaults.
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
`Population(GatedLIFNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~434K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps
