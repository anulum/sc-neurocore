# LiquidTimeConstantNeuron

**Module:** `sc_neurocore.neurons.models.ltc`
**Reference:** Hasani et al. 2021 (NeurIPS)
**Family:** Integrate-and-fire (input-adaptive time constant)
**State variables:** `x` (hidden state)

## Equations

$$\tau(x,I) = \tau_{base} \cdot \sigma(w_\tau \cdot I + b)$$
$$f(x,I) = \tanh(w_x \cdot x + w_{in} \cdot I)$$
$$x(t+1) = x(t) + \frac{dt}{\tau} \cdot (-x + f)$$

Spike: $x \geq V_\theta$, hard reset $x \to 0$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_base` | 10.0 | Base time constant |
| `w_tau` | -0.5 | Input → tau coupling |
| `w_x` | 0.8 | Self-coupling weight |
| `w_in` | 1.0 | Input weight |
| `v_threshold` | 1.0 | Spike threshold |

## Behaviour

- **Input-adaptive tau:** Time constant changes with input via sigmoid.
  Larger input → faster dynamics (w_tau < 0).
- **Sharp transition:** I ∈ [4, 4.5] is the critical range. Below:
  x settles to ~0.999 (subthreshold). Above: spikes every step.
- **tanh saturation:** f_target saturates at ±1, so x cannot exceed 1.0
  unless driven past the tanh ceiling.
- **NeurIPS 2021:** Designed for continuous-time sequence modelling.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, silent, subthreshold settle, spikes, sharp transition, tau input-dependent, tanh target, stability, reset, deterministic |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **14** | |
