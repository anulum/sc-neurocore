# GalvesLocherbachNeuron

**Module:** `sc_neurocore.neurons.models.galves_locherbach`
**Reference:** Galves & Löcherbach 2013
**Family:** Stochastic (point process)
**State variables:** `v` (membrane potential — accumulator, not ODE)

## Equations

$$V(t) = \gamma \cdot V(t-1) + w_{\text{input}}$$
$$P(\text{spike}) = \sigma\bigl(s \cdot (V - V_\theta)\bigr) \cdot dt$$

where $\sigma$ is the logistic sigmoid, $\gamma$ is decay, $s$ is steepness.

No ODE — purely probabilistic spiking with leaky integration.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `decay` | 0.95 | Voltage decay factor per step |
| `threshold_rate` | 0.5 | Centre of sigmoid (half-max firing) |
| `steepness` | 5.0 | Sigmoid sharpness |
| `dt` | 1.0 | Time step |

## Behaviour

- **Stochastic:** No deterministic threshold — spike probability is sigmoid
  of voltage. High steepness ≈ hard threshold.
- **Leaky accumulator:** Voltage decays by factor `decay` each step,
  accumulates weighted input.
- **Reset on spike:** v → v_rest after each spike.
- **Point process:** Mathematically rigorous stochastic neural model
  from probability theory (not biophysics).

## Infrastructure Pipeline

```
GalvesLocherbachNeuron
├── step(weighted_input) → int {0,1} (stochastic)
├── Population: PoissonInput(weight=1.0, rate=500Hz)
├── Verilog: sigmoid LUT + LFSR, ~40 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, stochastic spiking, rate increase, sigmoid probability, decay, reset on spike, stability, reset, custom steepness, low drive |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **14** | |
