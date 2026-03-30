# NeuroGridNeuron

**Module:** `sc_neurocore.neurons.models.neurogrid`
**Reference:** Boahen 2014
**Family:** Hardware (analog neuromorphic, 2-compartment)
**State variables:** `v_s` (soma voltage), `v_d` (dendrite voltage)

## Equations

**Dendrite (passive integrator):**
$$\tau_d \frac{dV_d}{dt} = -(V_d - V_r) + I - g_c(V_d - V_s)$$

**Soma (EIF with dendritic coupling):**
$$\tau_s \frac{dV_s}{dt} = -(V_s - V_r) + \Delta_T \exp\left(\frac{V_s - V_\theta}{\Delta_T}\right) + g_c(V_d - V_s)$$

Spike: $V_s \geq V_{peak} = 20$ mV, reset $V_s \to V_{reset}$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_s` | 20.0 | Soma time constant (ms) |
| `tau_d` | 50.0 | Dendrite time constant (ms) |
| `g_c` | 0.5 | Inter-compartment coupling conductance |
| `delta_t` | 2.0 | Exponential spike slope (mV) |
| `v_threshold` | -50.0 | EIF rheobase voltage (mV) |
| `v_peak` | 20.0 | Spike detection ceiling (mV) |
| `v_reset` | -65.0 | Post-spike reset (mV) |
| `dt` | 0.1 | Integration step (ms) |

## Behaviour

- **2-compartment:** Dendrite integrates synaptic input, soma generates
  spikes via exponential IF mechanism. Models the Neurogrid analog chip.
- **Dendritic filtering:** Slow dendrite (tau_d=50 ms) smooths input
  before it reaches the soma.
- **EIF soma:** Exponential term creates sharp spike initiation
  near threshold, with clipped exponent to prevent overflow.
- **Analog neuromorphic:** Models subthreshold analog computation
  as implemented in the Neurogrid mixed-signal VLSI chip.

## Infrastructure Pipeline

```
NeuroGridNeuron
├── step(current) → int {0,1}
├── Population: works
├── Verilog: 2 compartments + exp LUT, ~60 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 9 | construction, step binary, subthreshold, spikes, 2 compartments, dendritic integration, stability, reset, deterministic |
| Network | 1 | Population |
| Analysis | 1 | spike_count |
| **Total** | **11** | |
