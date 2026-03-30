# MorrisLecarNeuron

**Module:** `sc_neurocore.neurons.models.morris_lecar`
**Reference:** Morris & Lecar 1981
**Family:** Conductance-based (2D oscillator)
**State variables:** `v` (voltage), `w` (K activation)

## Equations

$$C_m \frac{dV}{dt} = -g_{Ca} m_\infty(V)(V-E_{Ca}) - g_K w(V-E_K) - g_L(V-E_L) + I$$
$$\frac{dw}{dt} = \lambda(V)(w_\infty(V) - w)$$

$$m_\infty(V) = \frac{1}{2}(1 + \tanh((V-V_1)/V_2))$$
$$w_\infty(V) = \frac{1}{2}(1 + \tanh((V-V_3)/V_4))$$
$$\lambda(V) = \phi \cosh((V-V_3)/(2V_4))$$

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_ca` | 4.0 | Ca conductance |
| `g_k` | 8.0 | K conductance |
| `g_l` | 2.0 | Leak conductance |
| `c_m` | 20.0 | Membrane capacitance |
| `phi` | 1/15 | K time-scale factor |
| `v1`–`v4` | varied | tanh half-activation/slope params |
| `dt` | 0.1 | Integration step (ms) |

## Behaviour

- **Type-II excitability:** Oscillation onset via subcritical Hopf bifurcation.
  Firing rate jumps discontinuously at threshold.
- **Non-monotonic f-I:** Rate peaks at intermediate I, decreases at high I
  (depolarisation block).
- **Ca-K oscillator:** Ca (instantaneous m_inf) depolarises, K (slow w) repolarises.
- **Canonical 2D model:** After HH, the most-studied conductance model.
  Used extensively in bifurcation analysis.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 12 | construction, step binary, subthreshold, spikes, Type-II non-monotonic, tanh activation, w recovery, lambda positive, stability, bounded, reset, deterministic |
| Network | 2 | Population, spikes |
| Analysis | 1 | spike_count |
| **Total** | **15** | |
