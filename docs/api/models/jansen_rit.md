# JansenRitUnit

**Module:** `sc_neurocore.neurons.models.jansen_rit`
**Reference:** Jansen & Rit 1995
**Family:** Neural mass (EEG generation)
**State variables:** `y0`–`y5` (6 ODEs: 3 populations × 2 states)

## Equations

3 coupled populations (pyramidal, excitatory interneuron, inhibitory interneuron),
each with a second-order linear operator + sigmoid nonlinearity:

$$\ddot{y}_0 = Aa \sigma(y_1 - y_2) - 2a\dot{y}_0 - a^2 y_0$$
$$\ddot{y}_1 = Aa(p + C_2\sigma(C_1 y_0)) - 2a\dot{y}_1 - a^2 y_1$$
$$\ddot{y}_2 = Bb C_4\sigma(C_3 y_0) - 2b\dot{y}_2 - b^2 y_2$$

Output: $\text{EEG}(t) = y_1(t) - y_2(t)$ (pyramidal PSP).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `a_exc` | 3.25 | Excitatory amplitude (mV) |
| `b_exc` | 22.0 | Inhibitory amplitude (mV) |
| `a_rate` | 100.0 | Excitatory rate (s⁻¹) |
| `b_rate` | 50.0 | Inhibitory rate (s⁻¹) |
| `c` | 135.0 | Connectivity constant |
| `e0` | 2.5 | Half of max firing rate |
| `v0` | 6.0 | Sigmoid midpoint (mV) |
| `r` | 0.56 | Sigmoid steepness |
| `dt` | 0.001 | Integration step (s) |

## Behaviour

- **EEG output:** Returns continuous voltage (y1-y2), not binary spikes.
  This is a mean-field model of a cortical column.
- **Alpha rhythm:** p_ext=220 produces ~10 Hz oscillation (alpha band).
- **Three regimes:** Low p → fixed point, medium p → alpha oscillation,
  high p → saturated oscillation.
- **Deterministic:** No noise in standard formulation.

## Infrastructure Pipeline

```
JansenRitUnit
├── step(p_ext) → float (EEG voltage)
├── Population: works (no spike output)
├── Verilog: 6 state regs + 3 sigmoid LUTs, ~200 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 11 | construction, step returns float, oscillation, bounded, zero drive stable, 6 states, sigmoid, drive effect, stability (6 vars), reset, deterministic |
| Network | 1 | Population |
| **Total** | **12** | |
