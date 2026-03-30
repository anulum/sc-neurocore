# LearnableNeuronModel

**Module:** `sc_neurocore.neurons.models.lnm`
**Reference:** Jahns et al. 2025
**Family:** Integrate-and-fire (fully learnable)
**State variables:** `v` (voltage)

## Equations

$$v(t) = \alpha \cdot v(t-1) + \beta \cdot I(t) + \gamma \cdot \sigma(v(t-1))$$

where $\sigma(v) = 1/(1+\exp(-s(v-c)))$.

Spike: $v \geq V_\theta$, hard reset $v \to 0$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.9 | Voltage decay (trainable) |
| `beta` | 0.1 | Input scaling (trainable) |
| `gamma` | 0.05 | Nonlinear feedback (trainable) |
| `f_slope` | 5.0 | Sigmoid steepness |
| `f_shift` | 0.5 | Sigmoid centre |
| `v_threshold` | 1.0 | Spike threshold |

## Behaviour

- **Fully trainable:** All 3 core params (alpha, beta, gamma) are
  differentiable — designed for gradient-based SNN optimisation.
- **Nonlinear feedback:** gamma * sigmoid(v) adds voltage-dependent
  self-excitation. gamma=0 reduces to linear LIF.
- **Hard reset:** v → 0 on spike.
- **Deterministic:** No stochastic element.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, silent, spikes, rate increase, alpha effect, beta effect, gamma=0 linear, stability, reset, deterministic |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **14** | |
