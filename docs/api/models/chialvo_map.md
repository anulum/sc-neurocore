# ChialvoMapNeuron

**Module:** `sc_neurocore.neurons.models.chialvo_map`
**Reference:** Chialvo 1995
**Family:** Map-based (discrete-time)
**State variables:** `x` (fast), `y` (slow recovery)

## Equations

$$x_{n+1} = x_n^2 \cdot e^{y_n - x_n} + k + I$$
$$y_{n+1} = a \cdot y_n - b \cdot x_n + c$$

Spike when $x_n \geq x_\theta$ (upward crossing).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `x` | 0.0 | Fast variable |
| `y` | 0.0 | Slow recovery |
| `a` | 0.89 | Recovery decay |
| `b` | 0.6 | Recovery coupling to x |
| `c` | 0.28 | Recovery drive |
| `k` | 0.04 | Intrinsic excitability |
| `x_threshold` | 1.0 | Spike threshold |

## Behaviour

- **Intrinsically excitable:** Spikes without input (k=0.04). Adding
  moderate current can SUPPRESS spiking (stabilisation of fixed point).
- **Map neuron:** Discrete-time, single exp() per step. Fast.
- **exp overflow fixed:** Uses `safe_exp()` from `utils.numerics`.

## Infrastructure Pipeline

```
ChialvoMapNeuron
├── step(current: float) → int {0,1}
├── reset() → x=0, y=0
├── In Population: scalar current
├── In Network: all stimuli, monitors, projections
├── Analysis: all spike_stats
├── SC encoding: rate coding
├── Verilog: compilable (1 exp LUT + multiply, ~40 LUTs)
└── Rust NetworkRunner: supported
```

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | 225 Ksteps/s | Not measured |
| Network (20 neurons, 500ms) | ~200 Kneuron-steps/s | Expected ~40× |

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 6 | construction, step binary, intrinsic spiking, state finite, safe_exp overflow, reset |
| Network | 3 | Population, spikes, Projection |
| Analysis | 3 | firing_rate, spike_count, ISI |
| **Total** | **12** | |

See `tests/test_model_chialvo_map.py`. Numerical fix: safe_exp for x²·exp(y-x).
