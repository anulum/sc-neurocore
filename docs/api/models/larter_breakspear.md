# LarterBreakspearNeuron

**Module:** `sc_neurocore.neurons.models.larter_breakspear`
**Reference:** Breakspear, Terry & Friston 2003
**Family:** Neural mass (ion-channel-based)
**State variables:** `v` (voltage), `w` (K recovery), `z` (slow adaptation)

## Equations

$$\frac{dV}{dt} = -I_{Ca} - I_{Na} - I_K - I_L + I_{ext} + C_{coupling} + a_{ee}V$$
$$\frac{dW}{dt} = \phi \frac{m_K(V) - W}{\tau_K}$$
$$\frac{dZ}{dt} = b(V + 0.5 - Z)$$

Ion currents use tanh-based sigmoidal activation (not Boltzmann).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_ca` | 1.1 | Ca conductance |
| `g_na` | 6.7 | Na conductance |
| `g_k` | 2.0 | K conductance |
| `g_l` | 0.5 | Leak conductance |
| `phi` | 0.7 | K time-scale |
| `b` | 0.1 | Slow adaptation rate |
| `a_ee` | 0.36 | Self-excitation |
| `i_ext` | 0.3 | External drive |
| `dt` | 0.01 | Integration step |

## Behaviour

- **Whole-brain modelling:** Designed for The Virtual Brain (TVB) —
  each node represents a cortical region, not a single neuron.
- **Continuous output:** Returns voltage (float), not binary spikes.
- **Ion-channel kinetics:** Ca, Na, K, leak with tanh sigmoidal gating.
- **3 time-scales:** Fast (v), medium (w), slow (z).
- **Bounded oscillation:** v ∈ [-0.5, 0.5] for default params.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step returns float, oscillation, bounded, 3 state vars, coupling, sigmoid gates, stability, reset, deterministic |
| Network | 1 | Population |
| **Total** | **11** | |
