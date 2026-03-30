# PrescottNeuron

**Module:** `sc_neurocore.neurons.models.prescott`
**Reference:** Prescott et al. 2008
**Family:** Reduced model (2D, excitability classification)
**State variables:** `v`, `w`

## Equations

$$\frac{dV}{dt} = -g_f m_\infty(V)(V - E_f) - g_s w(V - E_s) - g_L(V - E_L) + I$$
$$\frac{dw}{dt} = \phi \frac{w_\infty(V) - w}{\tau_w}$$

$m_\infty(V) = \sigma(-(V+20)/15)$, $w_\infty(V) = \sigma(-(V-\beta_w)/\gamma_w)$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | −65.0 | Membrane voltage |
| `w` | 0.0 | Slow recovery variable |
| `g_fast` | 20.0 | Fast Na-like conductance |
| `g_slow` | 20.0 | Slow K conductance |
| `beta_w` | −21.0 | w-nullcline half-activation |
| `tau_w` | 100.0 | w time constant (ms) |
| `phi` | 0.15 | Time-scale ratio |

## Behaviour

- **Slow oscillator:** ISI ∼ 5000–10000 steps at default parameters.
  Fires spontaneously even at I=0.
- **Excitability types via beta_w:** Lower beta_w → easier oscillation (Type I-like).
  Higher beta_w → suppressed firing (Type III-like). beta_w=0 nearly silences the model.
- **Non-linear g_slow interaction:** g_slow affects dynamics non-monotonically
  due to interaction with the fast subsystem.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary, 2-var evolution, finite 50k, reset |
| Oscillations | 4 | spontaneous, slow ISI (>1000), rate vs current, V amplitude |
| Excitability | 3 | beta_w modulation, high beta_w suppression, tau_w effect |
| Parameters | 5 | g_slow dynamics, dt stability (3 values), upward crossing |
| Determinism | 1 | bit-exact |
| Network | 2 | population, spikes |
| Analysis | 2 | spike_count, consistency |
| **Total** | **22** | |
