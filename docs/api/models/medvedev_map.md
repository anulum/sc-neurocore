# MedvedevMapNeuron

**Module:** `sc_neurocore.neurons.models.medvedev_map`
**Reference:** Medvedev 2005
**Family:** Map-based (1D chaotic)
**State variables:** `x` (phase, mod 1)

## Equations

$$x_{n+1} = \begin{cases} \alpha x + I & x < \beta \\ \alpha(1-x) + I & x \geq \beta \end{cases} \mod 1$$

Spike: upward crossing of $x_\theta = 0.9$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 3.5 | Map expansion rate |
| `beta` | 0.5 | Piecewise branch point |
| `x_threshold` | 0.9 | Spike detection threshold |

## Behaviour

- **1D chaotic map:** alpha > 2 produces chaotic dynamics.
  Sensitive dependence on initial conditions.
- **mod 1 bounded:** x always in [0, 1) — no divergence.
- **Piecewise-monotone:** Below beta scales linearly,
  above beta folds (tent-map-like).
- **Very efficient:** Single multiply + mod per step.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step binary, silent, spikes, x bounded, piecewise branches, rate increase, chaotic sensitivity, stability, reset, deterministic |
| Network | 1 | Population |
| Analysis | 1 | spike_count |
| **Total** | **13** | |
