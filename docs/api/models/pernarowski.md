# PernarowskiNeuron

**Module:** `sc_neurocore.neurons.models.pernarowski`
**Reference:** Pernarowski 1994
**Family:** Burster (FHN-like, 3D)
**State variables:** `v`, `w`, `z`

## Equations

$$\frac{dV}{dt} = V - \frac{V^3}{3} - w - z + I$$
$$\frac{dw}{dt} = \epsilon_1 (V - \gamma w + \alpha)$$
$$\frac{dz}{dt} = \epsilon_2 (\beta (V + 0.7) - z)$$

Spike: upward crossing of $V_\theta$ ($V_{t} \geq \theta$ and $V_{t-1} < \theta$).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | −1.0 | Membrane voltage |
| `w` | 0.0 | Fast recovery variable |
| `z` | 0.0 | Ultra-slow adaptation |
| `alpha` | 0.1 | w-nullcline offset |
| `beta` | 0.5 | z-nullcline slope |
| `eps1` | 0.1 | Time-scale ratio for w |
| `eps2` | 0.001 | Time-scale ratio for z (100× slower than w) |
| `gamma` | 0.5 | w self-coupling |
| `v_threshold` | 0.5 | Detection threshold |
| `dt` | 0.1 | Time step |

## Behaviour

- **Spontaneous oscillation:** Model bursts even at I=0 (relaxation oscillator).
  ISI ≈ 290–300 steps at default parameters.
- **Depolarisation block:** At I≥2.0, oscillation ceases — V converges to a
  stable high fixed point. Only 0–1 spikes in 10k steps.
- **Three time scales:** V (fast, ~dt), w (intermediate, eps1=0.1),
  z (ultra-slow, eps2=0.001). The z variable modulates burst envelope.
- **Voltage bounded:** V stays within approximately [−2, 2] (cubic nullcline).
- **Near-constant ISI:** CV(ISI) < 0.05 for constant input in oscillatory regime.
- **Deterministic:** No stochastic element.

## Dynamic Regimes

| Current range | Regime | Description |
|---------------|--------|-------------|
| I ∈ [0, 1.0] | Oscillatory | Sustained bursting, 30+ spikes/10k steps |
| I ∈ [1.0, 1.5] | Transitional | Reduced rate, lengthening ISI |
| I ≥ 2.0 | Depolarisation block | ≤1 spike, V converges to fixed point |

## Infrastructure Pipeline

```
PernarowskiNeuron
├── step(current) → int {0,1} (deterministic)
├── Population: PoissonInput(weight=0.5, rate=200Hz)
├── Verilog: 3 Euler accumulators + cubic LUT, ~120 LUTs
└── Rust: supported (3 f64 state variables)
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary, 3-var evolution, finite 50k, reset |
| Oscillations | 5 | spontaneous, V bounded, ISI regularity, ISI range, upward-only |
| f–I curve | 3 | sustained oscillation, depolarisation block, rate modulation |
| Slow variables | 3 | z slower than w, eps2 effect, z bounded |
| Parameters | 5 | custom threshold, gamma/beta sensitivity, dt stability (3 values) |
| Determinism | 1 | bit-exact reproducibility |
| Network | 2 | population, spikes |
| Analysis | 2 | spike_count, consistency |
| **Total** | **27** | |

Key finding: eps2 controls ultra-slow z dynamics. At eps2=0.001 (default),
z evolves ~100× slower than w, shaping the burst envelope.


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~277K steps/s |
| Spikes (10K steps, I=5.0) | 1 |
| State stability (20K steps) | PASS |
| Rust parity | EXACT |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`PernarowskiNeuron()` instantiates with documented defaults.
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
`Population(PernarowskiNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**EXACT** — Python and Rust produce identical spike trains.

---

## Findings (measured 2026-04-04)

1. Throughput: ~277K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: EXACT
4. Numerical stability confirmed over 20K steps
