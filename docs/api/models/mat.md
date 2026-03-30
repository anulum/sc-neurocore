# MATNeuron

**Module:** `sc_neurocore.neurons.models.mat`
**Reference:** Kobayashi et al. 2009
**Family:** Integrate-and-fire (multi-timescale adaptive threshold)
**State variables:** `v` (voltage), `theta1` (fast threshold), `theta2` (slow threshold)

## Equations

$$\tau_m \frac{dV}{dt} = -(V - V_r) + R \cdot I$$
$$\theta_1 \leftarrow \theta_1 \cdot \exp(-dt/\tau_1)$$
$$\theta_2 \leftarrow \theta_2 \cdot \exp(-dt/\tau_2)$$

Effective threshold: $\theta = \theta_{base} + \theta_1 + \theta_2$.

On spike: $V \to V_{reset}$, $\theta_1 \leftarrow \theta_1 + h_1$, $\theta_2 \leftarrow \theta_2 + h_2$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v_threshold_base` | -50.0 | Base threshold (mV) |
| `tau_1` | 10.0 | Fast adaptation time (ms) |
| `tau_2` | 200.0 | Slow adaptation time (ms) |
| `h1` | 5.0 | Fast threshold increment (mV) |
| `h2` | 3.0 | Slow threshold increment (mV) |
| `tau_m` | 10.0 | Membrane time constant (ms) |

## Behaviour

- **Two adaptation time-scales:** Fast (10 ms) handles burst termination,
  slow (200 ms) handles long-term rate adaptation.
- **Dynamic threshold:** Effective threshold rises after each spike,
  producing spike-frequency adaptation.
- **Kobayashi 2009:** Won the INCF spike time prediction challenge —
  best-performing simple model for cortical neuron prediction.
- **Deterministic:** No stochastic element.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, subthreshold, spikes, threshold adaptation, two timescales, adaptation reduces rate, rate increase, stability, reset, deterministic |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **14** | |
