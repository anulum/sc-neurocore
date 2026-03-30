# InhibitoryLIFNeuron

**Module:** `sc_neurocore.neurons.models.ilif`
**Reference:** SC-NeuroCore 2025
**Family:** Integrate-and-fire (inhibitory trace)
**State variables:** `v` (voltage), `inh_trace` (inhibitory trace)

## Equations

$$v(t) = \alpha_m \cdot v(t-1) + I - w_{inh} \cdot \text{trace}(t)$$
$$\text{trace}(t) = \alpha_{inh} \cdot \text{trace}(t-1)$$

On spike: $v \to v_{reset}$, $\text{trace} \leftarrow \text{trace} + 1$.

$\alpha_m = \exp(-dt/\tau_m)$, $\alpha_{inh} = \exp(-dt/\tau_{inh})$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_m` | 10.0 | Membrane time constant (ms) |
| `tau_inh` | 5.0 | Inhibitory trace decay (ms) |
| `v_threshold` | 1.0 | Spike threshold |
| `inh_strength` | 0.5 | Post-spike inhibition weight |
| `dt` | 1.0 | Time step |

## Behaviour

- **Temporal coding:** Post-spike inhibitory trace suppresses re-firing
  for a learned duration, shaping spike timing.
- **Stronger inhibition = lower rate:** inh_strength controls the
  trade-off between rate and temporal precision.
- **Deterministic:** No stochastic element.
- **Precomputed alphas:** Exponential decay factors computed at init.

## Infrastructure Pipeline

```
InhibitoryLIFNeuron
├── step(current) → int {0,1}
├── Population: PoissonInput(weight=0.5, rate=500Hz)
├── Verilog: 2 multiply-accumulate + compare, ~20 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 12 | construction, step binary, subthreshold, spikes, rate increase, trace increase, trace decay, inhibition reduces rate, alpha precomputed, stability, reset, deterministic |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **15** | |
