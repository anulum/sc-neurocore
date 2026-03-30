# GutkinErmentroutNeuron

**Module:** `sc_neurocore.neurons.models.gutkin_ermentrout`
**Reference:** Gutkin & Ermentrout 1998
**Family:** Conductance-based (minimal, 2D)
**State variables:** `v` (voltage), `n` (K activation)

## Equations

$$\frac{dV}{dt} = -g_{Na} m_\infty(V)(V-E_{Na}) - g_K n(V-E_K) - g_L(V-E_L) + I$$
$$\tau_n \frac{dn}{dt} = n_\infty(V) - n$$

m is instantaneous (persistent Na, no inactivation).
Spike: upward crossing of $V_\theta = -20$ mV.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_na` | 20.0 | Persistent Na conductance |
| `g_k` | 10.0 | K conductance |
| `g_l` | 8.0 | Leak conductance |
| `dt` | 0.05 | Integration step (ms) |

## Behaviour

- **Minimal 2D conductance:** Only 2 variables (v, n). m is instantaneous.
- **Persistent Na:** No inactivation → enables Type-I excitability
  (continuous f-I onset near threshold).
- **Deterministic:** No stochastic element.

## Infrastructure Pipeline

```
GutkinErmentroutNeuron
├── step(current) → int {0,1} (threshold crossing)
├── Population: PoissonInput(weight=100, rate=500Hz)
├── Verilog: 2 sigmoid LUTs + 2 channels, ~80 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step binary, subthreshold, spikes, rate increase, n gating, persistent Na (no m state), stability, reset, deterministic |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **13** | |
