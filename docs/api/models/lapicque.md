# LapicqueNeuron

**Module:** `sc_neurocore.neurons.models.lapicque`
**Reference:** Lapicque 1907
**Family:** Integrate-and-fire (classical)
**State variables:** `v` (voltage)

## Equations

$$\tau \frac{dV}{dt} = -(V - V_r) + R \cdot I$$

Spike: $V \geq V_\theta$, hard reset $V \to V_{reset}$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 20.0 | Membrane time constant (ms) |
| `resistance` | 1.0 | Membrane resistance |
| `v_threshold` | 1.0 | Spike threshold |
| `v_reset` | 0.0 | Post-spike reset |
| `dt` | 1.0 | Integration step |

## Behaviour

- **The original IF:** Lapicque 1907 — the first mathematical neuron model.
  Simple RC circuit with threshold.
- **Analytical rheobase:** I_rh = V_θ / R. Below rheobase, v settles to
  steady state R·I < V_θ. Above, periodic spiking.
- **Deterministic:** Fully deterministic Euler integration.
- **Hard reset:** v → v_reset (not subtract-reset).
- **Simplest conductance-free model:** No gating, no adaptation, no noise.

## Infrastructure Pipeline

```
LapicqueNeuron
├── step(current) → int {0,1}
├── Population: PoissonInput(weight=2.0, rate=500Hz)
├── Verilog: MAC + compare, ~10 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 12 | construction, step binary, subthreshold, spikes, rheobase, rate increase, voltage clamp, hard reset, stability, reset, deterministic, custom tau |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **15** | |
