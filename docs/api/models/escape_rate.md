# EscapeRateNeuron

**Module:** `sc_neurocore.neurons.models.escape_rate`
**Reference:** Gerstner 2000
**Family:** Stochastic (escape noise)
**State variables:** `v` (voltage)

## Equations

$$\tau_m \frac{dV}{dt} = -(V - V_r) + R \cdot I$$
$$\rho(V) = \rho_0 \exp\left(\frac{V - V_\theta}{\Delta u}\right)$$

Spike: Bernoulli with $p = \rho(V) \cdot dt$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rho_0` | 0.001 | Base escape rate |
| `delta_u` | 3.0 | Noise width (mV) |
| `v_threshold` | -50.0 | Nominal threshold (mV) |
| `tau_m` | 10.0 | Membrane time constant (ms) |

## Behaviour

- **Stochastic threshold:** No hard threshold — spike probability increases
  exponentially near V_θ. Soft threshold with Boltzmann-like noise.
- **safe_exp applied:** Prevents overflow at extreme voltages.
- **Low base rate:** ρ₀=0.001 → needs V near threshold for significant P(spike).

## Infrastructure Pipeline

```
EscapeRateNeuron
├── step(current) → int {0,1} (stochastic)
├── Population: PoissonInput(weight=50, rate=500Hz)
├── Verilog: exp LUT + LFSR for Bernoulli, ~50 LUTs
└── Rust: supported
```

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | 75.5 Ksteps/s | Not measured |

Slower due to np.random.random() per step.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 7 | construction, step binary, stochastic spiking, rate increases, safe_exp, state finite, reset |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **10** | |

safe_exp fix applied preventively.
