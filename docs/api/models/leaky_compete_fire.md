# LeakyCompeteFireNeuron

**Module:** `sc_neurocore.neurons.models.leaky_compete_fire`
**Reference:** Oster, Douglas & Liu 2009
**Family:** Winner-take-all (multi-unit)
**State variables:** `v` (list of voltages, one per unit)

## Equations

$$\tau \frac{dV_i}{dt} = -V_i + I_i$$

Spike: $V_i \geq V_\theta \Rightarrow V_i \to 0$, $V_j \leftarrow \max(0, V_j - w_{inh})$ for $j \neq i$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_units` | 4 | Number of competing units |
| `tau` | 10.0 | Membrane time constant |
| `v_threshold` | 1.0 | Spike threshold |
| `w_inh` | 0.5 | Lateral inhibition weight |
| `dt` | 1.0 | Time step |

## Behaviour

- **Winner-take-all:** Strongest-driven unit fires and suppresses
  all others via lateral inhibition.
- **Multi-unit output:** `step()` returns `list[int]` of length `n_units`.
- **Scalar broadcast:** Single current value applied to all units.
- **Non-negative:** Voltage clamped to ≥ 0 after inhibition.
- **Deterministic:** Same inputs → same WTA outcome.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step returns list, scalar broadcast, WTA dominance, lateral inhibition, no negative v, equal inputs, custom n_units, stability, reset, deterministic |
| Network | 1 | Population |
| **Total** | **12** | |
