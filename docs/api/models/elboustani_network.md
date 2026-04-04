# ElBoustaniNetwork

**Module:** `engine/src/neurons/population.rs`
**Reference:** El Boustani & Bhatt, J Comput Neurosci 26:313, 2009
**Family:** E/I mean-field with NMDA-mediated bistability
**State variables:** `r_e` (excitatory rate), `r_i` (inhibitory rate), `s` (NMDA gating)

---

## Biological Context

The El Boustani network extends the standard E/I mean-field by separating fast (AMPA) and slow (NMDA) excitatory recurrence. The slow NMDA component provides the positive feedback needed for persistent activity — a neural correlate of working memory.

Key features:
- **Dual excitatory pathways**: fast AMPA (j_ampa) + slow NMDA (j_nmda)
- **NMDA bistability**: s builds slowly with E activity, decays with tau_s ~ 100 ms
- **E/I balance**: inhibitory population stabilises the circuit
- **3 coupled ODEs**: r_e, r_i, s
- **Working memory regime**: high j_nmda allows persistent activity after stimulus removal

---

## Equations

$$\tau_e \frac{dr_e}{dt} = -r_e + \phi(J_{ampa} r_e + J_{nmda} s - J_{ei} r_i + I)$$
$$\tau_i \frac{dr_i}{dt} = -r_i + \phi(J_{ie} r_e - J_{ii} r_i)$$
$$\tau_s \frac{ds}{dt} = -s + \gamma \cdot r_e \cdot (1 - s)$$

where $\phi(z) = \max(0, z - \theta)$ is a threshold-linear transfer function.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `j_ampa` | 0.1 | Fast E→E coupling (AMPA) |
| `j_nmda` | 0.5 | Slow E→E coupling (NMDA) |
| `j_ei` | 0.8 | I→E coupling |
| `j_ie` | 0.5 | E→I coupling |
| `j_ii` | 0.2 | I→I coupling |
| `gamma` | 0.641 | NMDA saturation rate |
| `tau_e` | 20.0 ms | Excitatory time constant |
| `tau_i` | 10.0 ms | Inhibitory time constant |
| `tau_s` | 100.0 ms | NMDA decay time constant |
| `threshold` | 0.0 | Transfer function threshold |
| `dt` | 0.1 ms | Integration time step |

### Dynamical Regimes

- **Low j_nmda**: standard E/I dynamics, no persistent activity
- **High j_nmda**: bistability — stimulus triggers persistent activity that outlasts the input
- **Very high j_nmda**: runaway excitation (pathological, requires strong inhibition)

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/population.rs` |
| PyO3 wrapper | Yes (state: r_e, r_i, s) |
| NetworkRunner wired | `NeuronVariant::ElBoustani` |
| `create_neuron("ElBoustaniNetwork")` | Yes |
| `supported_models()` | Includes "ElBoustaniNetwork" |
| STRONG tests | 10 |
| Benchmark | `elboustani_100k_steps`: **6.05 ms** (60.5 ns/step), i5-11600K |
