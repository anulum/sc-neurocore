# BrunelNetwork

**Module:** `engine/src/neurons/population.rs`
**Reference:** Brunel, J Comput Neurosci 8:183, 2000
**Family:** Balanced E/I mean-field with threshold-linear transfer
**State variables:** `r_e` (excitatory rate), `r_i` (inhibitory rate)

---

## Biological Context

The Brunel balanced network model captures the dynamics of a cortical circuit with recurrent excitation and inhibition. The balance between E and I determines the regime: asynchronous irregular (AI, the default cortical state), synchronous regular (SR), or synchronous irregular (SI).

Key features:
- **E/I balance**: J_ei (I→E inhibition) counteracts J_ee (E→E excitation)
- **Threshold-linear transfer**: phi(x) = max(0, x - threshold)
- **Multiple regimes**: AI, SR, SI accessible via parameter tuning
- **2 coupled ODEs**: fast implementation

---

## Equations

$$\tau_e \frac{dr_e}{dt} = -r_e + \phi(J_{ee} r_e - J_{ei} r_i + I)$$
$$\tau_i \frac{dr_i}{dt} = -r_i + \phi(J_{ie} r_e - J_{ii} r_i)$$

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/population.rs` |
| PyO3 wrapper | Yes (state: r_e, r_i) |
| NetworkRunner wired | `NeuronVariant::Brunel` |
| `create_neuron("BrunelNetwork")` | Yes |
| `supported_models()` | Includes "BrunelNetwork" |
| STRONG tests | 10 |
| Benchmark | `brunel_100k_steps`: **3.48 ms** (34.8 ns/step), i5-11600K |
