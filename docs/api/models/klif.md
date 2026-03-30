# KLIFNeuron

**Module:** `sc_neurocore.neurons.models.klif`
**Reference:** SC-NeuroCore (AI-optimised variant)
**Family:** Integrate-and-fire (learnable)
**State variables:** `v` (voltage)

## Equations

$$v(t) = \alpha \cdot v(t-1) + k \cdot I(t)$$

Spike: $v \geq V_\theta$, hard reset $v \to 0$.
$\alpha = \exp(-dt/\tau)$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 1.0 | Learnable input scaling factor |
| `tau` | 10.0 | Membrane time constant |
| `v_threshold` | 1.0 | Spike threshold |
| `v_reset` | 0.0 | Post-spike reset |
| `dt` | 1.0 | Time step |

## Behaviour

- **Single learnable parameter:** k scales input current — trainable
  via STE or surrogate gradients.
- **Hard reset:** v → 0 on spike (not subtract-reset).
- **Deterministic:** Identical input → identical output.
- **Simpler than GatedLIF:** One gate instead of two.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step binary, subthreshold, spikes, k effect, alpha precomputed, hard reset, stability, reset, deterministic |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **13** | |
