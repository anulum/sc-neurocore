# GIFPopulationNeuron

**Module:** `sc_neurocore.neurons.models.gif_population`
**Reference:** Mensi et al. 2012
**Family:** Integrate-and-fire (generalised, stochastic)
**State variables:** `v` (voltage), `eta` (adaptation current)

## Equations

$$\tau_m \frac{dV}{dt} = -(V - V_r) - \eta + I$$
$$\eta \leftarrow \eta \cdot \exp(-dt/\tau_\eta)$$
$$h(V) = \lambda_0 \exp\left(\frac{V - \theta}{\Delta_V}\right)$$
$$P(\text{spike}) = 1 - \exp(-h \cdot dt)$$

On spike: $V \to V_{reset}$, $\eta \leftarrow \eta + \eta_{inc}$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `theta` | -50.0 | Baseline threshold (mV) |
| `tau_m` | 20.0 | Membrane time constant (ms) |
| `tau_eta` | 100.0 | Adaptation decay time (ms) |
| `delta_v` | 2.0 | Escape-rate sharpness (mV) |
| `lambda_0` | 0.001 | Base hazard rate (ms⁻¹) |
| `eta_increment` | 5.0 | Spike-triggered adaptation (mV) |
| `dt` | 0.5 | Integration step (ms) |

## Behaviour

- **Escape-rate threshold:** Stochastic spiking with exponential hazard.
  Softer than hard threshold — P(spike) increases smoothly with V.
- **Spike-frequency adaptation:** Each spike adds `eta_increment` to eta,
  which decays exponentially with `tau_eta`. Reduces firing rate over time.
- **Population-level:** Designed for mean-field population models.
- **Stochastic:** Two identical neurons with same input will fire differently.

## Infrastructure Pipeline

```
GIFPopulationNeuron
├── step(current) → int {0,1} (stochastic)
├── Population: PoissonInput(weight=30, rate=500Hz)
├── Verilog: exp LUT + LFSR + adaptation register, ~70 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step binary, subthreshold, spikes, stochastic, adaptation increase, adaptation decay, rate increase, stability, reset |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **13** | |
