# COBALIFNeuron

**Module:** `sc_neurocore.neurons.models.coba_lif`
**Reference:** Destexhe et al. 2001
**Family:** Integrate-and-Fire (conductance-based)
**State variables:** `v` (voltage), `g_e` (excitatory conductance), `g_i` (inhibitory conductance)

## Equations

$$C_m \frac{dV}{dt} = -g_L(V-E_L) - g_e(V-E_e) - g_i(V-E_i) + I$$
$$\frac{dg_e}{dt} = -g_e / \tau_e, \quad \frac{dg_i}{dt} = -g_i / \tau_i$$

Spike: $V \geq V_\theta \Rightarrow V \to V_\text{reset}$.
Conductances injected via `delta_ge`, `delta_gi` parameters.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | -65.0 | Membrane voltage (mV) |
| `g_e`, `g_i` | 0.0 | Synaptic conductances (nS) |
| `c_m` | 200.0 | Membrane capacitance (pF) |
| `g_l` | 10.0 | Leak conductance (nS) |
| `e_l` | -65.0 | Leak reversal (mV) |
| `e_e` | 0.0 | Excitatory reversal (mV) |
| `e_i` | -80.0 | Inhibitory reversal (mV) |
| `tau_e` | 5.0 | Excitatory decay (ms) |
| `tau_i` | 10.0 | Inhibitory decay (ms) |
| `v_threshold` | -50.0 | Spike threshold (mV) |
| `dt` | 0.1 | Timestep (ms) |

## Behaviour

- **Conductance-based:** Synaptic current depends on membrane voltage
  (driving force V-E_rev). More biophysical than current-based LIF.
- **Extra step() parameters:** `delta_ge` and `delta_gi` inject conductance
  increments (modeling synaptic events). In Population, these default to 0
  — only current injection via PoissonInput is used.
- **High C_m:** 200 pF requires substantial current (≥500) for spiking.

## Infrastructure Pipeline

```
COBALIFNeuron
├── step(current, delta_ge=0, delta_gi=0) → int {0,1}
├── reset() → v=e_l, g_e=0, g_i=0
├── In Population: scalar current only (delta_ge/gi unused)
│   For conductance injection: use custom Network step or direct neuron access
├── In Network: PoissonInput (weight=500, rate=500Hz)
├── Analysis: all spike_stats
├── SC encoding: rate coding
├── Verilog: compilable (2 exp for conductance decay, ~60 LUTs)
└── Rust NetworkRunner: supported (scalar current interface)
```

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | 477 Ksteps/s | Not measured |
| Network (20, 500ms) | ~400 Kneuron-steps/s | Expected ~40× |
| Spiking threshold | I ≥ 500 | — |

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 9 | construction, step binary, subthreshold, spikes, g_e decay, delta_ge injection, delta_gi injection, state finite, reset |
| Network | 3 | Population, spikes, Projection |
| Analysis | 3 | firing_rate, spike_count, ISI |
| **Total** | **15** | |

See `tests/test_model_coba_lif.py`. No bugs found.
