# ExpIFNeuron

**Module:** `sc_neurocore.neurons.models.expif`
**Reference:** Fourcaud-Trocmé et al. 2003
**Family:** Integrate-and-fire (exponential)
**State variables:** `v` (voltage)

## Equations

$$\tau \frac{dV}{dt} = -(V - V_r) + \Delta_T \exp\left(\frac{V - V_{rh}}{\Delta_T}\right) + I$$

Spike: $V \geq V_\theta$, reset to $V_{reset}$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v_rest` | -65.0 | Resting potential (mV) |
| `v_reset` | -68.0 | Post-spike reset (mV) |
| `v_threshold` | -50.0 | Hard spike threshold (mV) |
| `v_rh` | -55.0 | Rheobase voltage — centre of exponential escape (mV) |
| `delta_t` | 2.0 | Sharpness of exponential term (mV) |
| `tau` | 20.0 | Membrane time constant (ms) |
| `dt` | 0.1 | Integration step (ms) |

## Behaviour

- **Exponential escape:** voltage near `v_rh` triggers exponential runaway,
  producing a sharp spike. Larger `delta_t` = softer onset.
- **No adaptation:** no w variable — pure spike generator.
- **np.clip guard:** exp argument clipped to [-20, 20] preventing overflow.
- **Type-II-like:** sharp threshold, no continuous f-I onset.

## Infrastructure Pipeline

```
ExpIFNeuron
├── step(current) → int {0,1}
├── Population: PoissonInput(weight=20, rate=500Hz)
├── Verilog: exp LUT, ~30 LUTs (simpler than AdEx)
└── Rust: supported via NeuronVariant
```

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~220 Ksteps/s | Not measured |

Lightweight — single exp + clip per step.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, subthreshold, spikes, rate increase, exponential escape, exp clipping, negative extreme, numerical stability, reset, custom params |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **14** | |
