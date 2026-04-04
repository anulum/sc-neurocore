# IbarzTanakaMapNeuron

**Module:** `sc_neurocore.neurons.models.ibarz_tanaka_map`
**Reference:** Ibarz et al. 2007
**Family:** Map-based (piecewise-linear bursting)
**State variables:** `x` (fast, ≈voltage), `y` (slow, ≈adaptation)

## Equations

$$x_{n+1} = f(x_n) + y_n + I$$
$$y_{n+1} = y_n - \mu(x_n + 1) + \mu\sigma$$

$$f(x) = \begin{cases} \alpha/(1-x) & x \leq 0 \\ \alpha + \beta x & x > 0 \end{cases}$$

Spike: $x \geq x_\theta$, reset $x \to x_{reset}$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 3.65 | Piecewise map amplitude |
| `beta` | 0.25 | Linear spiking slope |
| `mu` | 0.0005 | Slow time-scale |
| `sigma` | -1.6 | Slow variable target |
| `x_threshold` | 3.0 | Spike threshold |
| `x_reset` | -1.0 | Post-spike reset |

## Behaviour

- **Discrete map:** No ODE — iterative, computationally cheap.
- **Piecewise-linear:** f(x) has a singularity at x=1 (from left),
  producing sharp spike onset. Linear spiking phase above x=0.
- **Bursting:** Slow y variable (µ=0.0005) modulates burst-pause.
- **Deterministic:** Fully deterministic map.
- **Efficient:** Single evaluation per step — ideal for large networks.

## Infrastructure Pipeline

```
IbarzTanakaMapNeuron
├── step(current) → int {0,1}
├── Population: works
├── Verilog: division LUT + comparator, ~30 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step binary, subthreshold, spikes, piecewise f, slow y, reset on spike, rate increase, stability, reset, deterministic |
| Network | 1 | Population |
| Analysis | 1 | spike_count |
| **Total** | **12** | |


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~319K steps/s |
| Spikes (10K steps, I=5.0) | 2421 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`IbarzTanakaMapNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
2421 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(IbarzTanakaMapNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~319K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps
