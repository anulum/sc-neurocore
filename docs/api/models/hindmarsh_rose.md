# HindmarshRoseNeuron

**Module:** `sc_neurocore.neurons.models.hindmarsh_rose`
**Reference:** Hindmarsh & Rose 1984
**Family:** Oscillator / Burster (3D, chaotic)
**State variables:** `x` (fast, ≈voltage), `y` (fast, ≈recovery), `z` (slow, ≈adaptation)

## Equations

$$\frac{dx}{dt} = y - x^3 + bx^2 - z + I$$
$$\frac{dy}{dt} = 1 - 5x^2 - y$$
$$\frac{dz}{dt} = r\bigl(s(x - x_r) - z\bigr)$$

Spike: upward crossing of $x_\theta = 1.0$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `b` | 3.0 | Quadratic coefficient (excitability) |
| `r` | 0.001 | Slow time-scale (smaller = slower bursts) |
| `s` | 4.0 | Slow variable coupling |
| `x_rest` | -1.6 | Resting x value |
| `x_threshold` | 1.0 | Spike detection threshold |
| `dt` | 0.1 | Integration step |

## Behaviour

- **Chaotic bursting:** For I ∈ [2, 5], alternates between rapid spike
  bursts and silent pauses. The slow z variable controls the burst envelope.
- **3 dynamical regimes:** Quiescent (I<2), bursting (2<I<5), tonic (I>5).
- **Canonical burster:** Most-studied bursting model in computational
  neuroscience. Used for chaos analysis, synchronisation studies.
- **Bounded orbit:** x, y, z remain bounded for physiological I.

## Infrastructure Pipeline

```
HindmarshRoseNeuron
├── step(current) → int {0,1} (threshold crossing)
├── Population: PoissonInput(weight=5, rate=200Hz)
├── Verilog: polynomial (x³, x²), ~60 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, subthreshold, spikes, bursting (ISI ratio), 3 state vars, slow z, rate increase, stability, bounded orbit, reset |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **14** | |


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~247K steps/s |
| Spikes (10K steps, I=5.0) | 156 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`HindmarshRoseNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
156 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(HindmarshRoseNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~247K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps
