# McCullochPittsNeuron

**Module:** `sc_neurocore.neurons.models.mcculloch_pitts`
**Reference:** McCulloch & Pitts 1943
**Family:** Binary (stateless threshold)
**State variables:** None

## Equations

$$y = \begin{cases} 1 & \text{if } \sum w_i x_i \geq \theta \\ 0 & \text{otherwise} \end{cases}$$

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `theta` | 1.0 | Threshold |

## Behaviour

- **The first mathematical neuron** (1943). Founding model of computational neuroscience.
- **Stateless:** No membrane, no history — pure combinational logic.
- **Logic gates:** theta=2 → AND, theta=1 → OR, theta=0.5 with negative weight → NOT.
- **Deterministic:** Identical input → identical output always.
- **reset() is no-op:** No state to reset.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 12 | construction, step binary, below/at/above threshold, negative input, stateless, custom theta, reset noop, deterministic, AND gate, OR gate |
| Network | 1 | Population |
| **Total** | **13** | |
