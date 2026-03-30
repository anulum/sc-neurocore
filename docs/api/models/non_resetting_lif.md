# NonResettingLIFNeuron

**Module:** `sc_neurocore.neurons.models.non_resetting_lif`
**Reference:** Kobayashi et al. 2009, Jolivet et al. 2004
**Family:** Integrate-and-fire (non-resetting, adaptive threshold)
**State variables:** `v` (voltage), `theta` (dynamic threshold)

## Equations

$$\tau_m \frac{dV}{dt} = -(V - V_r) + R \cdot I$$
$$\tau_\theta \frac{d\theta}{dt} = -(\theta - \theta_r)$$

Spike: $V \geq \theta$, then $\theta \leftarrow \theta + \Delta_\theta$.

**Critically: $V$ does NOT reset.** Only the threshold jumps up.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v_rest` | -65.0 | Resting potential (mV) |
| `theta_rest` | -50.0 | Baseline threshold (mV) |
| `delta_theta` | 5.0 | Threshold jump on spike (mV) |
| `tau_m` | 10.0 | Membrane time constant (ms) |
| `tau_theta` | 50.0 | Threshold relaxation time (ms) |
| `r_m` | 1.0 | Membrane resistance |
| `dt` | 0.1 | Integration step (ms) |

## Behaviour

- **No voltage reset:** Unlike standard LIF, voltage continues its
  natural trajectory after spike. Only the threshold jumps up by
  delta_theta, preventing immediate re-firing.
- **Self-limiting:** Repeated spiking accumulates theta increases,
  naturally reducing rate over time (adaptation).
- **Theta decays:** Between spikes, theta relaxes back to theta_rest
  with time constant tau_theta.
- **aMAT variant:** Closely related to the MAT family (Kobayashi 2009).
  Differs in the absence of voltage reset — preserves voltage information
  across spikes.

## Infrastructure Pipeline

```
NonResettingLIFNeuron
├── step(current) → int {0,1}
├── Population: works
├── Verilog: LIF + threshold register, ~20 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step binary, subthreshold, spikes, no voltage reset, theta increase, theta decay, stability, reset, deterministic |
| Network | 1 | Population |
| Analysis | 1 | spike_count |
| **Total** | **12** | |
