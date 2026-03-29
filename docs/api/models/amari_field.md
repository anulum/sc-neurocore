# AmariNeuralField

**Module:** `sc_neurocore.neurons.models.amari_field`
**Reference:** Amari 1977
**Family:** Population / Neural Field
**State variables:** `u` (N-dimensional activation field)

## Equations

$$\tau \frac{du_i}{dt} = -u_i + \sum_j w(|i-j|) \, f(u_j) \, dx + I_i$$

Kernel (Mexican hat):
$$w(x) = A \, e^{-a|x|} - B \, e^{-b|x|}$$

Activation: $f(u) = \max(0, u)$

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n` | 64 | Number of field nodes |
| `tau` | 10.0 | Time constant |
| `a_exc` | 1.5 | Excitatory amplitude A |
| `a_width` | 1.0 | Excitatory spatial decay a |
| `b_inh` | 0.75 | Inhibitory amplitude B |
| `b_width` | 2.0 | Inhibitory spatial decay b |
| `dx` | 0.5 | Spatial resolution |
| `dt` | 0.5 | Timestep |

## Behaviour

- **Neural field:** This is a population-level model (N=64 nodes by default),
  NOT a single-neuron model. Each "neuron" instance is an entire 1D field.
- **Mexican hat connectivity:** Excitatory centre, inhibitory surround.
  Supports bump attractor formation for working memory and decision-making.
- **Continuous activation:** `step()` returns mean activation (float), not
  binary spike. When used in Population, the return value is clipped to {0,1}.
- **FFT convolution:** The spatial interaction is computed via FFT for O(N log N)
  performance.

## Infrastructure Pipeline

```
AmariNeuralField
├── step(current: NDArray) → float (mean activation)
├── reset() → zeros field
├── In Population: scalar input broadcasts to all N nodes
│   └── Return value clipped to int8 {0,1} via Population.step_all()
├── In Network: compatible with PoissonInput, StepCurrent, TimedArray
├── Analysis: firing_rate on binary spike train (after Population clip)
└── NOT directly compatible with:
    ├── SC bitstream encoding (float output, not spike)
    ├── STDP (no spike timing)
    └── Rust NetworkRunner (population model, not single neuron)
```

## Wiring Plan

```
PoissonInput(weight=20, rate=1000Hz)
    ↓ scalar current per field instance
Population(AmariNeuralField, n=K)
    ↓ K fields, each with N=64 internal nodes
    ↓ step() returns float, clipped to {0,1}
SpikeMonitor
    ↓ binary spike trains
Analysis toolkit (firing_rate, spike_count)
```

**Note:** For meaningful analysis, use the field directly (not through
Population). Access `neuron.u` for the full N-dimensional state.

## Performance

| Backend | N=64 field | 1000 steps | Notes |
|---------|-----------|------------|-------|
| Python (NumPy) | ~0.5 ms/step | ~0.5 s | FFT-based, vectorised |
| Rust | N/A | N/A | Not supported (population model) |

## Test Coverage

See `tests/test_model_amari_field.py` (11 tests):
- Isolation: construction, custom size, step return type, localised bump
  formation, Mexican hat kernel sign, state finiteness, reset, scalar broadcast
- Network: Population creation, network run, field state after drive

**Production bug found and fixed:** Population.step_all() overflowed int8
when model returned float > 127. Fixed with `min(max(int(raw), 0), 1)` clip.
