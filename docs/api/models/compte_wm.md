# CompteWMNeuron

**Module:** `sc_neurocore.neurons.models.compte_wm`
**Reference:** Compte et al. 2000
**Family:** Biophysical (working memory / prefrontal cortex)
**State variables:** `v`, `s_ampa`, `s_nmda`, `x_nmda`, `s_gaba`

## Equations

$$C_m \frac{dV}{dt} = -I_L - I_{AMPA} - I_{NMDA} - I_{GABA} + I_{\text{ext}}$$
$$I_{NMDA} = g_{NMDA} \cdot B(V) \cdot s_{NMDA} \cdot (V - E_e)$$
$$B(V) = \frac{1}{1 + [Mg^{2+}]/3.57 \cdot e^{-0.062V}}$$

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | -70.0 | Voltage (mV) |
| `g_nmda` | 0.165 | NMDA conductance |
| `tau_nmda` | 100.0 | NMDA decay (ms) — slow, supports persistent activity |
| `mg` | 1.0 | Mg²⁺ concentration (mM) |
| `v_threshold` | -50.0 | Spike threshold (mV) |
| `dt` | 0.1 | Timestep (ms) |

## Behaviour

- **Working memory:** Slow NMDA kinetics (τ=100 ms) support persistent
  activity after stimulus removal — the neural basis of working memory.
- **Mg²⁺ block:** Voltage-dependent NMDA unblock creates positive feedback:
  depolarisation → more NMDA current → more depolarisation (bistability).
- **Self-inhibition:** Spike triggers s_gaba += 1.0 (autaptic GABA).
- **Extra step() param:** `spike_in=True` injects AMPA + NMDA input.

## Infrastructure Pipeline

```
CompteWMNeuron
├── step(current, spike_in=False) → int {0,1}
├── reset() → v=e_l, all conductances zero
├── In Population: scalar current (spike_in unused)
├── In Network: PoissonInput (weight=2.0, rate=500Hz)
├── Analysis: all spike_stats
├── Verilog: compilable (1 exp for Mg block, 3 exp for decay, ~80 LUTs)
└── Rust: supported (scalar interface)
```

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | 262 Ksteps/s | Not measured |
| Network (20, 500ms) | ~220 Kneuron-steps/s | Expected ~30× |
| Spiking threshold | I ≥ 1.0 | — |

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 9 | construction, step binary, subthreshold, spikes, NMDA gating via spike_in, Mg block voltage dependence, GABA self-inhibition, state finite, reset |
| Network | 3 | Population, spikes, Projection |
| Analysis | 3 | firing_rate, spike_count, ISI |
| **Total** | **15** | |

See `tests/test_model_compte_wm.py`. No bugs found.
