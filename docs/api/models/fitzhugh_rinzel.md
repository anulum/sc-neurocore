# FitzHughRinzelNeuron

**Module:** `sc_neurocore.neurons.models.fitzhugh_rinzel`
**Reference:** FitzHugh 1976, Rinzel 1987
**Family:** Oscillator / Burster (3D)
**State variables:** `v` (voltage), `w` (fast recovery), `y` (ultra-slow modulation)

## Equations

$$\frac{dv}{dt} = v - \frac{v^3}{3} - w + y + I$$
$$\frac{dw}{dt} = \delta(a + v - bw)$$
$$\frac{dy}{dt} = \mu(c - v - dy)$$

Spike: upward crossing of $v_\theta = 1.0$ (no reset — oscillatory model).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | -1.0 | Initial voltage |
| `w` | -0.5 | Initial fast recovery |
| `y` | 0.0 | Initial slow variable |
| `a` | 0.7 | w-nullcline offset |
| `b` | 0.8 | w-nullcline slope |
| `c` | -0.775 | y-nullcline offset |
| `d` | 1.0 | y-nullcline slope |
| `delta` | 0.08 | Fast time-scale (≈ FHN ε) |
| `mu` | 0.0001 | Ultra-slow time-scale — controls burst envelope |
| `dt` | 0.1 | Integration step (ms) |

## Behaviour

- **Bursting:** Third variable y modulates oscillation amplitude on an
  ultra-slow time-scale (µ=0.0001), producing burst-pause patterns.
- **FHN core:** v-w dynamics identical to FitzHughNagumo; y acts as a
  slowly drifting bias current.
- **Oscillatory band:** Spikes in I∈[0.5,1.2] (same as FHN core).
- **Bounded orbit:** All three variables stay bounded.

## Infrastructure Pipeline

```
FitzHughRinzelNeuron
├── step(current) → int {0,1} (threshold crossing)
├── Population: PoissonInput(weight=2.0, rate=200Hz)
├── Verilog: 3 state variables, ~60 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step binary, subthreshold, spikes, 3 state vars, slow drift, stability, bounded, reset, custom mu |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **13** | |
