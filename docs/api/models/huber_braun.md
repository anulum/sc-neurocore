# HuberBraunNeuron

**Module:** `sc_neurocore.neurons.models.huber_braun`
**Reference:** Braun, Huber et al. 1998
**Family:** Conductance-based (cold receptor, temperature-dependent)
**State variables:** `v`, `a_sd` (slow depolarising), `a_sr` (slow repolarising)

## Equations

$$\frac{dV}{dt} = -g_{sd} a_{sd}(V-E_{sd}) - g_{sr} a_{sr}(V-E_{sr}) - g_L(V-E_L) + I + \eta\xi(t)$$
$$\tau_{sd} \frac{da_{sd}}{dt} = \sigma_{sd}(V) - a_{sd}$$
$$\tau_{sr} \frac{da_{sr}}{dt} = \sigma_{sr}(V) - a_{sr}$$

where $\sigma$ are sigmoid activation functions and $\xi(t)$ is Gaussian noise.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_sd` | 1.5 | Slow depolarising conductance |
| `g_sr` | 0.4 | Slow repolarising conductance |
| `g_l` | 0.1 | Leak conductance |
| `tau_sd` | 10.0 | SD time constant (ms) |
| `tau_sr` | 20.0 | SR time constant (ms) |
| `eta` | 0.012 | Noise amplitude |
| `dt` | 0.1 | Integration step (ms) |

## Behaviour

- **Cold receptor:** Models temperature-sensitive neurons in the skin.
  Oscillation regime depends on temperature (mapped to conductance ratios).
- **Default params:** Produce a single spike then settle to depolarised
  equilibrium (~+46 mV). Sustained oscillation requires parameter tuning.
- **Stochastic:** Gaussian noise (eta > 0) can trigger stochastic resonance.
- **No fast Na inactivation:** Simplified model — lacks repolarisation
  mechanism for sustained spiking in default regime.

## Infrastructure Pipeline

```
HuberBraunNeuron
├── step(current) → int {0,1} (threshold crossing)
├── Population: works
├── Verilog: 2 sigmoid LUTs + noise LFSR, ~80 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, initial spike, sd gating, sr gating, noise present, no noise deterministic, stability, gating bounded, reset, depolarised equilibrium |
| Network | 1 | Population |
| Analysis | 1 | spike_count |
| **Total** | **13** | |
