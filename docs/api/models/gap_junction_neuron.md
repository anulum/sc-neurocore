# GapJunctionNeuron

**Module:** `engine/src/neurons/misc.rs`
**Reference:** Connors & Long, Annu Rev Neurosci 27:393, 2004
**Family:** LIF with electrical synapse coupling
**State variables:** `v` (membrane potential)

---

## Biological Context

Gap junctions (electrical synapses) allow direct electrical current flow between neurons via connexin channels. Unlike chemical synapses, they are:

- **Bidirectional**: current flows in both directions
- **Fast**: no synaptic delay
- **Low-pass**: preferentially transmit slow voltage changes

Found extensively in:
- **Inferior olive** — synchronised climbing fibre discharge
- **Cortical PV+ interneuron networks** — gamma oscillation synchrony
- **Thalamic reticular nucleus** — spindle wave propagation
- **Retinal ganglion cell networks** — correlated firing

The gap junction current is proportional to the voltage difference between coupled cells:
I_gap = g_gap * (V_neighbor - V)

In the single-neuron pipeline, the input current represents the mean neighbour voltage.

Key features:
- **Electrical coupling**: g_gap * (V_neighbor - V)
- **LIF dynamics**: spike-and-reset with refractory period
- **Tonic drive**: optional I_tonic for intrinsic excitability
- **1 ODE**: fast computation

---

## Equations

$$C_m \frac{dV}{dt} = -g_L(V - E_L) + g_{gap}(V_{neighbor} - V) + I_{tonic}$$

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `c_m` | 1.0 | Membrane capacitance |
| `g_l` | 0.1 | Leak conductance |
| `e_l` | -65.0 mV | Leak reversal potential |
| `g_gap` | 0.05 | Gap junction conductance |
| `i_tonic` | 0.0 | Tonic depolarising current |
| `v_threshold` | -50.0 mV | Spike threshold |
| `v_reset` | -65.0 mV | Post-spike reset |
| `refractory` | 2.0 ms | Refractory period |
| `dt` | 0.1 ms | Integration time step |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/misc.rs` |
| PyO3 wrapper | Yes (state: v) |
| NetworkRunner wired | `NeuronVariant::GapJunction` |
| `create_neuron("GapJunctionNeuron")` | Yes |
| `supported_models()` | Includes "GapJunctionNeuron" |
| STRONG tests | 10 |
| Benchmark | `gap_junction_100k_steps`: **6.28 ms** (62.8 ns/step), i5-11600K |
