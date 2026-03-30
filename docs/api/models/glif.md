# GLIFNeuron

**Module:** `sc_neurocore.neurons.models.glif`
**Reference:** Teeter et al. 2018, Nature Communications (Allen Institute)
**Family:** Integrate-and-fire (generalised, 5-level hierarchy)
**State variables:** `v`, `theta`, `i_asc1`, `i_asc2`

## Equations

$$\tau_m \frac{dV}{dt} = -(V - V_r) + R \cdot I + I_{asc1} + I_{asc2}$$
$$\tau_\theta \frac{d\theta}{dt} = \theta_\infty - \theta + a_\theta(V - V_r)$$
$$I_{asc,j} \leftarrow I_{asc,j} \cdot \exp(-dt/\tau_{asc,j})$$

Spike: $V \geq \theta$, then $V \to V_{reset}$, $\theta \leftarrow \theta + \Delta_\theta$,
$I_{asc,j} \leftarrow I_{asc,j} + r_j$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_m` | 10.0 | Membrane time constant (ms) |
| `tau_theta` | 100.0 | Threshold adaptation time (ms) |
| `tau_asc1` | 10.0 | Fast ASC decay (ms) |
| `tau_asc2` | 200.0 | Slow ASC decay (ms) |
| `delta_theta` | 2.0 | Threshold jump on spike (mV) |
| `a_theta` | 0.01 | Voltage-dependent threshold adaptation |
| `r_asc1` | 1.0 | Fast ASC increment on spike |
| `r_asc2` | 0.5 | Slow ASC increment on spike |

## Behaviour

- **GLIF5 hierarchy:** Full 5-level Allen Institute model.
  Level 1: LIF. Level 2: +reset rules. Level 3: +instantaneous threshold.
  Level 4: +threshold adaptation. Level 5: +after-spike currents.
- **Dynamic threshold:** theta increases on each spike and relaxes back.
  Produces spike-frequency adaptation.
- **Two ASC time-scales:** Fast (10ms) + slow (200ms) after-spike currents
  shape the inter-spike interval distribution.
- **Deterministic:** No stochastic element.

## Infrastructure Pipeline

```
GLIFNeuron
├── step(current) → int {0,1}
├── Population: PoissonInput(weight=30, rate=500Hz)
├── Verilog: 4 state registers + 2 exp decays, ~120 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, subthreshold, spikes, rate increase, threshold adaptation, ASC increase, ASC decay, stability, reset, deterministic |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **14** | |
