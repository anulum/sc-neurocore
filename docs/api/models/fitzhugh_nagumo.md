# FitzHughNagumoNeuron

**Module:** `sc_neurocore.neurons.models.fitzhugh_nagumo`
**Reference:** FitzHugh 1961, Nagumo et al. 1962
**Family:** Oscillator (2D qualitative)
**State variables:** `v` (voltage), `w` (recovery)

## Equations

$$\frac{dv}{dt} = v - \frac{v^3}{3} - w + I$$
$$\frac{dw}{dt} = \varepsilon(v + a - bw)$$

Spike: upward crossing of $v_\theta = 1.0$ (no reset — oscillatory model).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | -1.0 | Initial voltage |
| `w` | -0.5 | Initial recovery |
| `a` | 0.7 | w-nullcline offset |
| `b` | 0.8 | w-nullcline slope |
| `epsilon` | 0.08 | Time-scale separation (slow w dynamics) |
| `v_threshold` | 1.0 | Spike detection threshold |
| `dt` | 0.1 | Integration step (ms) |

## Behaviour

- **Oscillatory:** No hard reset — v traces a limit cycle through the cubic
  nullcline. Spike = upward crossing of v_threshold.
- **Type-II excitability:** Discontinuous onset of oscillation at critical I.
  Below threshold I, model settles to stable fixed point.
- **Bounded orbit:** v and w remain bounded for physiological I (<5).
- **Canonical form:** Simplified Hodgkin-Huxley, widely used in dynamical
  systems analysis. Phase plane plots particularly informative.

## Infrastructure Pipeline

```
FitzHughNagumoNeuron
├── step(current) → int {0,1} (threshold crossing)
├── Population: PoissonInput(weight=2.0, rate=200Hz)
├── Verilog: polynomial + multiply, ~40 LUTs
├── Phase plane: nullcline overlay in Studio
└── Rust: supported via NeuronVariant
```

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~400 Ksteps/s | Not measured |

Lightweight — no exp, no gating variables. Two multiplies per step.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, subthreshold, spikes, oscillatory band (I∈[0.5,1.2]), oscillatory dynamics, w recovery, numerical stability, bounded orbit, reset, custom epsilon |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **14** | |
