# TUMNetwork

**Module:** `engine/src/neurons/population.rs`
**Reference:** Tsodyks, Uziel & Markram, J Neurosci 20:RC50, 2000
**Family:** Mean-field rate model with short-term synaptic plasticity (STP)
**State variables:** `r` (population rate), `x` (available resources), `u` (release probability)

---

## Biological Context

The Tsodyks-Uziel-Markram (TUM) model couples a population-level firing rate equation with short-term synaptic plasticity (STP). This captures two key biological mechanisms:

- **Synaptic depression**: repeated presynaptic activity depletes vesicle resources (x decreases)
- **Synaptic facilitation**: repeated activity increases release probability (u increases)

The effective synaptic strength is `u * x * J`, which dynamically modulates the recurrent coupling. This produces transient amplification (initial burst when resources are full) followed by adaptation (rate drops as x depletes), a hallmark of cortical responses to sustained stimuli.

Key features:
- **Depression + facilitation**: two opposing STP mechanisms
- **Transient amplification**: strong initial response, weaker sustained response
- **3 coupled ODEs**: rate + depression + facilitation
- **Rate-based**: fast computation, no spike resolution

---

## Equations

$$\tau \frac{dr}{dt} = -r + \phi(u \cdot x \cdot J \cdot r + I)$$
$$\frac{dx}{dt} = \frac{1 - x}{\tau_d} - u \cdot x \cdot r$$
$$\frac{du}{dt} = \frac{U - u}{\tau_f} + U \cdot (1 - u) \cdot r$$

where $\phi(z) = \max(0, z - \theta)$ is a threshold-linear transfer function.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `j` | 5.0 | Base synaptic strength |
| `u_base` | 0.2 | Baseline release probability (U) |
| `tau` | 10.0 ms | Rate time constant |
| `tau_d` | 200.0 ms | Depression recovery time constant |
| `tau_f` | 50.0 ms | Facilitation decay time constant |
| `threshold` | 0.0 | Transfer function threshold |
| `gain_phi` | 1.0 | Transfer function gain |
| `dt` | 0.1 ms | Integration time step |

### Dynamical Regimes

- **Depression-dominated** (tau_d >> tau_f): strong transient, weak sustained — typical of depressing cortical synapses
- **Facilitation-dominated** (tau_f >> tau_d): weak initial, building sustained — typical of facilitating synapses (e.g., mossy fibre)
- **Balanced**: intermediate dynamics, both effects visible

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/population.rs` |
| PyO3 wrapper | Yes (state: r, x, u) |
| NetworkRunner wired | `NeuronVariant::TUM` |
| `create_neuron("TUMNetwork")` | Yes |
| `supported_models()` | Includes "TUMNetwork" |
| STRONG tests | 10 |
| Benchmark | `tum_100k_steps`: **15.63 ms** (156.3 ns/step), i5-11600K |
