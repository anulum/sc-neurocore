# CazellesMapNeuron

**Module:** `sc_neurocore.neurons.models.cazelles_map`
**Reference:** Cazelles et al. 2001
**Family:** Map-based (discrete-time bursting)
**State variables:** `x` (fast membrane), `y` (slow modulation)

## Equations

$$x_{n+1} = a \cdot x_n (1 - x_n) - y_n + I$$
$$y_{n+1} = y_n + \epsilon (x_n - \sigma)$$

Spike when $x \geq x_\theta$. x clipped to [-2, 2].

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `x` | 0.1 | Fast variable (membrane-like) |
| `y` | 0.0 | Slow modulation variable |
| `a` | 3.8 | Logistic map parameter |
| `epsilon` | 0.01 | Slow variable coupling |
| `sigma` | 0.5 | Slow variable equilibrium |
| `x_threshold` | 0.9 | Spike threshold on x |

## Behaviour

- **Map neuron:** Discrete-time, no ODE integration. Each `step()` is one
  iteration of the map. Fast and numerically safe (no exp).
- **Bursting:** Slow y modulates fast x dynamics. When y builds up, x is
  suppressed → inter-burst interval. When y decays, x oscillates → burst.
- **Logistic dynamics:** Fast map $f(x) = ax(1-x)$ produces chaotic-like
  oscillations at a=3.8 (near chaotic regime of logistic map).

## Infrastructure Pipeline

```
CazellesMapNeuron
├── step(current: float) → int {0,1}
├── reset() → x=0.1, y=0.0
├── In Population: scalar current, standard interface
├── In Network: all stimuli and monitors
│   ├── PoissonInput (weight=0.3, rate=500Hz)
│   └── Projection compatible
├── Analysis: all spike_stats (bursting pattern)
├── SC encoding: spike train → rate coding
├── Verilog: easily compilable (no exp/log — multiply + clip)
│   Estimated ~20 LUTs per neuron
└── Rust NetworkRunner: supported
```

## Wiring Plan

```
PoissonInput(weight=0.3, rate=500Hz)
    ↓ small current pulses (map dynamics sensitive to input amplitude)
Population(CazellesMapNeuron, n=N)
    ↓ binary spike vector (burst pattern)
SpikeMonitor → spike_trains → analysis
```

## Performance

| Metric | Python (NumPy) | Rust engine |
|--------|---------------|-------------|
| Isolation | 303 Ksteps/s | Not measured |
| Network (20 neurons, 500ms) | ~250 Kneuron-steps/s | Expected ~40× faster |

Fast model — no transcendental functions.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 7 | construction, step binary, spikes, slow variable, x clipped, state finite, reset |
| Network | 3 | Population, spike production, Projection |
| Analysis | 3 | firing_rate, spike_count, ISI |
| **Total** | **13** | |

See `tests/test_model_cazelles_map.py`. No bugs found.
