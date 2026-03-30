# GatedLIFNeuron

**Module:** `sc_neurocore.neurons.models.gated_lif`
**Reference:** Yao et al. 2022 (NeurIPS)
**Family:** Integrate-and-fire (learnable gated)
**State variables:** `v` (voltage)

## Equations

$$v(t) = g_v \cdot v(t-1) + g_i \cdot I(t)$$

Spike: $v \geq V_\theta$, subtract-reset: $v \leftarrow v - V_\theta$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gate_v` | 0.9 | Voltage decay gate (learnable in training) |
| `gate_i` | 1.0 | Input scaling gate (learnable in training) |
| `v_threshold` | 1.0 | Spike threshold |
| `dt` | 1.0 | Time step |

## Behaviour

- **Learnable gates:** `gate_v` and `gate_i` are trainable parameters
  in SNN training frameworks. Replaces fixed decay constant.
- **Subtract-reset:** v -= V_θ on spike (preserves excess voltage).
- **Deterministic:** No stochastic element — identical input = identical output.
- **AI-optimised:** Designed for deep SNN training, not biophysics.

## Infrastructure Pipeline

```
GatedLIFNeuron
├── step(current) → int {0,1}
├── Population: PoissonInput(weight=0.5, rate=500Hz)
├── Verilog: 2 multiplies + compare, ~15 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, subthreshold, spikes, subtract reset, rate increase, gate_v effect, gate_i effect, stability, reset, deterministic |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **14** | |
