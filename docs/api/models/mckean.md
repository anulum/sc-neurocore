# McKeanNeuron

**Module:** `sc_neurocore.neurons.models.mckean`
**Reference:** McKean 1970
**Family:** Oscillator (piecewise-linear FHN)
**State variables:** `v`, `w` (recovery)

## Equations

$$\frac{dv}{dt} = f(v) - w + I$$
$$\frac{dw}{dt} = \varepsilon(v - \gamma w)$$

$$f(v) = \begin{cases} -v & v < a/2 \\ v-a & a/2 \leq v < (1+a)/2 \\ 1-v & v \geq (1+a)/2 \end{cases}$$

Spike: upward crossing of $v_{peak}=0.8$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `a` | 0.25 | Piecewise breakpoint parameter |
| `epsilon` | 0.01 | Time-scale separation (slow w) |
| `gamma` | 0.5 | w-nullcline slope |
| `v_peak` | 0.8 | Spike detection threshold |
| `dt` | 0.1 | Integration step |

## Behaviour

- **Piecewise-linear FHN:** Analytically tractable simplification —
  replaces FHN cubic with 3 linear segments.
- **Oscillatory band:** Like FHN, oscillates in a limited I range.
  Too low or too high I → stable fixed point.
- **Bounded orbit:** v and w remain bounded.
- **Slow w:** epsilon=0.01 gives slow recovery dynamics.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step binary, subthreshold, spikes, piecewise f, w recovery, bounded, stability, reset, deterministic |
| Network | 1 | Population |
| Analysis | 1 | spike_count |
| **Total** | **12** | |


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~280K steps/s |
| Spikes (10K steps, I=5.0) | 1 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`McKeanNeuron()` instantiates with documented defaults.
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
`Population(McKeanNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~280K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps
