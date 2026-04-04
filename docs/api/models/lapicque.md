# LapicqueNeuron

**Module:** `sc_neurocore.neurons.models.lapicque`
**Reference:** Lapicque 1907
**Family:** Integrate-and-fire (classical)
**State variables:** `v` (voltage)

## Equations

$$\tau \frac{dV}{dt} = -(V - V_r) + R \cdot I$$

Spike: $V \geq V_\theta$, hard reset $V \to V_{reset}$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 20.0 | Membrane time constant (ms) |
| `resistance` | 1.0 | Membrane resistance |
| `v_threshold` | 1.0 | Spike threshold |
| `v_reset` | 0.0 | Post-spike reset |
| `dt` | 1.0 | Integration step |

## Behaviour

- **The original IF:** Lapicque 1907 — the first mathematical neuron model.
  Simple RC circuit with threshold.
- **Analytical rheobase:** I_rh = V_θ / R. Below rheobase, v settles to
  steady state R·I < V_θ. Above, periodic spiking.
- **Deterministic:** Fully deterministic Euler integration.
- **Hard reset:** v → v_reset (not subtract-reset).
- **Simplest conductance-free model:** No gating, no adaptation, no noise.

## Infrastructure Pipeline

```
LapicqueNeuron
├── step(current) → int {0,1}
├── Population: PoissonInput(weight=2.0, rate=500Hz)
├── Verilog: MAC + compare, ~10 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 12 | construction, step binary, subthreshold, spikes, rheobase, rate increase, voltage clamp, hard reset, stability, reset, deterministic, custom tau |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **15** | |


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~758K steps/s |
| Spikes (10K steps, I=5.0) | 2000 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

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

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~758K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps
