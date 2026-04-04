# GradedSynapseNeuron

**Module:** `engine/src/neurons/misc.rs`
**Reference:** Roberts & Bush, J Comp Physiol A 185:549, 1999
**Family:** Non-spiking interneuron with graded transmitter release
**State variables:** `v` (membrane potential)

---

## Biological Context

Non-spiking interneurons communicate via graded changes in membrane potential rather than action potentials. Found in:

- **Retinal bipolar and amacrine cells** — graded transmission in retinal circuitry
- **C. elegans interneurons** — most C. elegans neurons are non-spiking
- **Crustacean stomatogastric ganglia** — pattern generators with graded synapses
- **Insect visual interneurons** — tonic graded signalling

The membrane potential follows passive RC dynamics. Transmitter release is a continuous sigmoid function of V, not all-or-nothing. A "spike" event is emitted when V crosses a threshold, representing a significant release transition for pipeline compatibility.

Key features:
- **Graded release**: sigmoid function of V, not binary
- **Saturation**: V clamped to [v_min, v_max]
- **Passive dynamics**: no regenerative sodium channels
- **1 ODE**: extremely fast computation

---

## Equations

$$C_m \frac{dV}{dt} = -g_L(V - E_L) + g_{in} \cdot I_{ext}$$

**Release function:**
$$\text{release}(V) = \frac{1}{1 + \exp(-(V - V_{half}) / k)}$$

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `c_m` | 1.0 | Membrane capacitance |
| `g_l` | 0.05 | Leak conductance |
| `e_l` | -60.0 mV | Leak reversal potential |
| `g_in` | 0.1 | Input conductance scaling |
| `v_half` | -40.0 mV | Release sigmoid half-activation |
| `k_release` | 5.0 | Release sigmoid slope |
| `v_threshold` | -35.0 mV | Pipeline "spike" threshold |
| `dt` | 0.1 ms | Integration time step |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/misc.rs` |
| PyO3 wrapper | Yes (state: v) |
| NetworkRunner wired | `NeuronVariant::GradedSynapse` |
| `create_neuron("GradedSynapseNeuron")` | Yes |
| `supported_models()` | Includes "GradedSynapseNeuron" |
| STRONG tests | 10 |
| Benchmark | `graded_synapse_100k_steps`: **4.66 ms** (46.6 ns/step), i5-11600K |
