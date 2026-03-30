# IntegerQIFNeuron

**Module:** `sc_neurocore.neurons.models.iqif`
**Reference:** Lo et al. 2021
**Family:** Integrate-and-fire (integer, FPGA-native)
**State variables:** `v` (integer voltage)

## Equations

$$V[t+1] = \max(V_{min},\ V[t] + (V[t]^2 \gg k) + I)$$

Spike: $V \geq V_\theta$, reset $V \to V_{reset}$.

All arithmetic is integer. The quadratic term $V^2 \gg k$ replaces
floating-point division — maps directly to FPGA shift register.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 6 | Right-shift for V² (controls quadratic gain) |
| `v_threshold` | 1024 | Spike threshold (integer) |
| `v_reset` | -1024 | Post-spike reset |
| `v_min` | -2048 | Voltage floor (prevents underflow) |

## Behaviour

- **Pure integer:** No floating-point — directly synthesisable to FPGA.
- **Quadratic nonlinearity:** V² >> k creates Type-I excitability.
  Larger k = more damped = fewer spikes.
- **High sensitivity:** Spikes at I=5 with default params.
  I=0 is exactly silent (V stays at 0).
- **Deterministic:** Fully deterministic integer map.
- **v_min clamp:** Prevents voltage underflow from negative reset.

## Infrastructure Pipeline

```
IntegerQIFNeuron
├── step(current: int) → int {0,1}
├── Population: works (integer current)
├── Verilog: multiply + shift + compare, ~15 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 12 | construction, step binary, silent at zero, spikes, integer type, bit shift, v_min clamp, reset on spike, rate increase, custom k, reset, deterministic |
| Network | 1 | Population |
| Analysis | 1 | spike_count |
| **Total** | **14** | |
