# EnergyLIFNeuron

**Module:** `sc_neurocore.neurons.models.energy_lif`
**Reference:** Fardet & Levina 2020
**Family:** Integrate-and-Fire (metabolic constraint)
**State variables:** `v` (voltage), `epsilon` (energy)

## Equations

$$\tau_m \frac{dV}{dt} = -(V-V_r) + \epsilon \cdot R \cdot I$$
$$\tau_\epsilon \frac{d\epsilon}{dt} = \epsilon_0 - \epsilon$$

Spike: $V \geq V_\theta$ AND $\epsilon > 0.1$.
On spike: $V \to V_{reset}$, $\epsilon \to \epsilon - \alpha$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epsilon` | 1.0 | Metabolic energy (dimensionless) |
| `tau_e` | 500.0 | Energy recovery time constant (ms) |
| `alpha` | 0.1 | Energy cost per spike |
| `epsilon_0` | 1.0 | Resting energy level |

## Behaviour

- **Metabolic gating:** Cannot spike when energy depleted (ε < 0.1).
- **Energy depletion:** Each spike costs α=0.1. At high firing rates
  energy drops, reducing effective resistance → lower excitability.
- **Recovery:** Without spiking, ε recovers to ε₀ with τ_e=500ms.

## Infrastructure Pipeline

```
EnergyLIFNeuron
├── step(current) → int {0,1}
├── Energy gates spiking (ε > 0.1 required)
├── Population, Network: PoissonInput(weight=30, rate=500Hz)
├── Verilog: compilable (~25 LUTs, adder + comparator + energy counter)
└── Rust: supported
```

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | 1.76 Msteps/s | Not measured |
| Network (20, 500ms) | ~1.4 Mneuron-steps/s | — |

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 9 | construction, step binary, subthreshold, spikes, energy depletes, energy recovers, energy gates spiking, energy nonneg, reset |
| Network | 3 | Population, spikes, Projection |
| Analysis | 2 | firing_rate, spike_count |
| **Total** | **14** | |

No bugs found. No transcendentals.
