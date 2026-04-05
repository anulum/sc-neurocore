# MyelinatedAxon

**Module:** `engine/src/neurons/misc.rs`
**Reference:** McIntyre, Richardson & Grill, J Neurophysiol 87:995, 2002
**Family:** Saltatory conduction segment (node + internode double-cable)
**State variables:** node.v (node membrane potential), v_inter (internode voltage), plus all NodeOfRanvier gating variables

---

## Biological Context

A myelinated axon segment consists of an active node of Ranvier (where ion channels cluster) coupled to a passive internode (myelinated segment) via paranodal seals. This is the fundamental unit of saltatory conduction.

The MRG 2002 double-cable model captures:
- **Active node**: full NodeOfRanvier model (Nav1.6 transient + persistent, Kv7)
- **Passive internode**: very low capacitance (myelin layers, ~0.001 µF/cm²), very low leak
- **Paranodal seal**: conductance between node and internode at the junction

The internode acts as a leaky cable that:
- Stores charge (very low capacitance → fast propagation)
- Provides current return path to the node
- Determines conduction velocity via its length and properties

---

## Equations

Node:
$$C_n \frac{dV_n}{dt} = I_{ionic}(V_n) + g_{para}(V_i - V_n) + I_{ext}$$

Internode:
$$C_i \frac{dV_i}{dt} = -g_{L,myelin}(V_i - E_L) + g_{para}(V_n - V_i)$$

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `c_inter` | 0.001 µF/cm² | Internode capacitance (myelin) |
| `g_l_myelin` | 0.001 mS/cm² | Myelin leak |
| `g_para` | 0.01 mS/cm² | Paranodal seal conductance |
| Node params | (see NodeOfRanvier) | Full MRG 2002 node |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/misc.rs` |
| PyO3 wrapper | Yes (state: v_inter) |
| NetworkRunner wired | `NeuronVariant::MyelinAxon` |
| `create_neuron("MyelinatedAxon")` | Yes |
| `supported_models()` | Includes "MyelinatedAxon" |
| STRONG tests | 10 |
| Benchmark | `myelinated_axon_1k_steps`: **1.26 ms** (1.26 µs/step), i5-11600K |
