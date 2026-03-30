# GolombFSNeuron

**Module:** `sc_neurocore.neurons.models.golomb_fs`
**Reference:** Golomb et al. 2007
**Family:** Conductance-based (fast-spiking interneuron)
**State variables:** `v`, `h` (Na inactivation), `n` (Kd activation), `p` (Kv3 activation)

## Equations

$$C_m \frac{dV}{dt} = -I_{Na} - I_{Kd} - I_{Kv3} - I_L + I_{ext}$$

where $I_{Na} = g_{Na} m_\infty^3 h (V-E_{Na})$, $I_{Kd} = g_{Kd} n^4 (V-E_K)$,
$I_{Kv3} = g_{Kv3} p^2 (V-E_K)$, $I_L = g_L(V-E_L)$.

m is instantaneous (no ODE). h, n, p follow first-order kinetics.

Spike: upward crossing of $V_\theta = -20$ mV.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_na` | 112.5 | Na conductance |
| `g_kd` | 225.0 | Delayed-rectifier K conductance |
| `g_kv3` | 150.0 | Kv3 K conductance (FS marker) |
| `g_l` | 0.25 | Leak conductance |
| `dt` | 0.01 | Sub-step size (ms), 10 sub-steps per call |

## Behaviour

- **Fast-spiking:** Kv3 channel enables narrow spikes and sustained
  high-frequency firing without adaptation — characteristic of
  PV+ cortical interneurons.
- **HH-type:** Full conductance-based model with Na, Kd, Kv3, leak.
- **10 sub-steps:** dt=0.01ms internal, 0.1ms effective per call.
- **No adaptation:** Rate scales monotonically with input.

## Infrastructure Pipeline

```
GolombFSNeuron
├── step(current) → int {0,1} (threshold crossing)
├── Population: PoissonInput(weight=10, rate=500Hz)
├── Verilog: 3 gating LUTs + 4 current channels, ~250 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step binary, subthreshold, spikes, fast-spiking, rate increase, Kv3 gating, gating bounded, stability, reset |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **13** | |
