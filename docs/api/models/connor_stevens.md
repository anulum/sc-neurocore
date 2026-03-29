# ConnorStevensNeuron

**Module:** `sc_neurocore.neurons.models.connor_stevens`
**Reference:** Connor & Stevens 1977
**Family:** Biophysical (Type-I excitability)
**State variables:** `v`, `m`, `h`, `n` (HH-like), `a`, `b` (A-type K⁺)

## Equations

HH-like Na⁺/K⁺ + A-type transient K⁺ current:
$$I_A = g_A \, a^3 \, b \, (V - E_A)$$

6 state variables. 100 sub-steps per `step()` call (dt=0.01 ms, loop 1/dt).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_a` | 47.7 | A-type K⁺ conductance |
| `e_a` | -75.0 | A-type reversal (mV) |
| `dt` | 0.01 | Sub-step timestep (ms) |
| `v_threshold` | 0.0 | Spike threshold (mV) |

## Behaviour

- **Type-I excitability:** Firing rate increases continuously from zero
  near threshold (saddle-node bifurcation). Arbitrarily low firing rates
  possible near rheobase.
- **A-type K⁺ current:** Delays spike onset, creates latency to first spike.
- **Very slow:** 100 sub-steps per network step. ~1K steps/s isolation.

## Infrastructure Pipeline

```
ConnorStevensNeuron
├── step(current) → int {0,1}
├── 100 sub-steps per call (dt=0.01ms)
├── Population, Network, SpikeMonitor: compatible
│   PoissonInput(weight=15, rate=500Hz)
├── Verilog: expensive (~300 LUTs, 100-cycle pipeline)
└── Rust: supported but slow
```

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | 1.1 Ksteps/s | Not measured |
| Network (5 neurons, 200ms) | ~150 neuron-steps/s | — |

**Slowest model in the library** — 100 sub-steps × 8 exp per sub-step.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 8 | construction, step binary, subthreshold, spikes, Type-I rate increase, A-type gating, numerical stability, reset |
| Network | 2 | Population, network spikes |
| Analysis | 1 | spike_count |
| **Total** | **11** | |

See `tests/test_model_connor_stevens.py`. No bugs found.
