# ErmentroutKopellPopulation

**Module:** `sc_neurocore.neurons.models.ermentrout_kopell_pop`
**Reference:** Montbrio, Pazo & Roxin 2015
**Family:** Population / Mean-field
**State variables:** `r` (mean firing rate), `v` (mean voltage)

## Equations

$$\tau \frac{dr}{dt} = \frac{\Delta}{\pi\tau} + 2rv$$
$$\tau \frac{dv}{dt} = v^2 + \bar\eta + I + J\tau r - (\pi\tau r)^2$$

Exact mean-field reduction of infinite QIF/theta neuron network.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r` | 0.1 | Population firing rate |
| `v` | -2.0 | Mean membrane potential |
| `tau` | 1.0 | Time constant |
| `delta` | 1.0 | Heterogeneity width (Cauchy) |
| `eta_bar` | -5.0 | Mean excitability |
| `j` | 15.0 | Coupling strength |
| `dt` | 0.01 | Timestep |

## Behaviour

- **Mean-field model:** Each "neuron" instance represents an ENTIRE
  population. `step()` returns firing rate r (float), not binary spike.
- **In Population:** r clipped to {0,1}. When r > 1, spike=1 (persistent).
  When r < 1, spike=0. Semantically different from spiking models.
- **Bistable:** With J=15, system has excitable and active fixed points.

## Infrastructure Pipeline

```
ErmentroutKopellPopulation
├── step(ext_input) → float (firing rate r)
├── In Population: clipped to {0,1} — persistent spike when r>1
├── Network: PoissonInput(weight=5, rate=500Hz)
├── Verilog: NOT directly compilable (v² requires wide multiply)
└── Rust: supported (scalar interface)
```

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | 973 Ksteps/s | Not measured |

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 6 | construction, step returns float, rate increases, rate nonneg, state finite, reset |
| Network | 3 | Population, network runs, field state |
| **Total** | **9** | |

No bugs found. No transcendentals.
