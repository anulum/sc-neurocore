# DestexheThalamicNeuron

**Module:** `sc_neurocore.neurons.models.destexhe_thalamic`
**Reference:** Destexhe 1993
**Family:** Biophysical (thalamocortical relay)
**State variables:** `v`, `h_na`, `n_k`, `m_t`, `h_t`

## Equations

Na⁺/K⁺ (HH-like) + low-threshold T-type Ca²⁺ current.
$$I_T = g_T \, m_T^2 \, h_T \, (V - E_{Ca})$$
5 sub-steps per `step()`. T-current produces post-inhibitory rebound spikes.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_t` | 2.0 | T-type Ca²⁺ conductance |
| `e_ca` | 120.0 | Ca²⁺ reversal (mV) |
| `dt` | 0.02 | Sub-step timestep (ms) |
| `v_threshold` | -20.0 | Spike threshold (mV) |

## Behaviour

- **Thalamocortical relay:** Switches between tonic (depolarised) and
  burst (hyperpolarised) modes. T-current de-inactivates during
  hyperpolarisation → rebound burst on release.
- **Single rebound spike:** At constant drive, produces one spike then
  settles near -23 mV (T-current inactivated). Needs periodic
  hyperpolarisation for repeated bursting.

## Infrastructure Pipeline

```
DestexheThalamicNeuron
├── step(current) → int {0,1}
├── 5 sub-steps per call
├── Population, Network: PoissonInput(weight=5, rate=500Hz)
├── Verilog: ~120 LUTs (5 Boltzmann + T-current)
└── Rust: supported
```

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | 32.5 Ksteps/s | Not measured |
| Network (10, 500ms) | ~25 Kneuron-steps/s | — |

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 6 | construction, step binary, rebound spike, T gating, numerical stability, reset |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **9** | |

No bugs found. tau guards (`max(tau, 0.1)`) prevent divide-by-zero.
