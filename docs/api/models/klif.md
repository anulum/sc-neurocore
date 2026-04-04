# KLIFNeuron

**Module:** `sc_neurocore.neurons.models.klif`
**Reference:** SC-NeuroCore (AI-optimised variant)
**Family:** Integrate-and-fire (learnable)
**State variables:** `v` (voltage)

## Equations

$$v(t) = \alpha \cdot v(t-1) + k \cdot I(t)$$

Spike: $v \geq V_\theta$, hard reset $v \to 0$.
$\alpha = \exp(-dt/\tau)$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 1.0 | Learnable input scaling factor |
| `tau` | 10.0 | Membrane time constant |
| `v_threshold` | 1.0 | Spike threshold |
| `v_reset` | 0.0 | Post-spike reset |
| `dt` | 1.0 | Time step |

## Behaviour

- **Single learnable parameter:** k scales input current — trainable
  via STE or surrogate gradients.
- **Hard reset:** v → 0 on spike (not subtract-reset).
- **Deterministic:** Identical input → identical output.
- **Simpler than GatedLIF:** One gate instead of two.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step binary, subthreshold, spikes, k effect, alpha precomputed, hard reset, stability, reset, deterministic |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **13** | |


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~325K steps/s |
| Spikes (10K steps, I=5.0) | 10000 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`KLIFNeuron()` instantiates with documented defaults.
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
`Population(KLIFNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~325K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps
