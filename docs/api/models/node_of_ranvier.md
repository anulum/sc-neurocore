# NodeOfRanvier

**Module:** `engine/src/neurons/misc.rs`
**Reference:** McIntyre, Richardson & Grill, J Neurophysiol 87:995, 2002
**Family:** Mammalian myelinated axon node (MRG 2002)
**State variables:** `v` (membrane potential), `m` (Nav1.6 transient activation), `h` (Nav1.6 inactivation), `p` (Nav1.6 persistent activation), `s` (Kv7 slow K activation)

---

## Biological Context

The node of Ranvier is the unmyelinated gap (~1 µm) between adjacent myelin segments where voltage-gated channels cluster at high density for saltatory conduction. The MRG 2002 model is the gold standard for mammalian nodal electrophysiology.

Key channel complement (distinct from generic HH):
- **Nav1.6 transient** (g_nat = 3000 mS/cm²): dominant nodal Na channel, fast m³h gating
- **Nav1.6 persistent** (g_nap = 5 mS/cm²): subthreshold amplification via p³ gating — lowers effective firing threshold, critical for saltatory propagation fidelity
- **Kv7/KCNQ slow K** (g_ks = 80 mS/cm²): membrane stabilisation, not fast repolarisation
- **No fast K (Kv1)**: Kv1 channels are paranodal/juxtaparanodal, not at the node itself

The persistent Na current is the key distinguishing feature — it provides subthreshold amplification that ensures reliable saltatory conduction.

---

## Equations

$$C_m \frac{dV}{dt} = -(I_{NaT} + I_{NaP} + I_{Ks} + I_L) + I_{ext}$$

$$I_{NaT} = g_{NaT} \cdot m^3 \cdot h \cdot (V - E_{Na})$$
$$I_{NaP} = g_{NaP} \cdot p^3 \cdot (V - E_{Na})$$
$$I_{Ks} = g_{Ks} \cdot s \cdot (V - E_K)$$

Gating uses Boltzmann steady-state + voltage-dependent time constant:
$$\frac{dx}{dt} = \frac{x_\infty(V) - x}{\tau_x(V)}$$

### Parameters (MRG 2002)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `c_m` | 2.0 µF/cm² | Nodal capacitance |
| `g_nat` | 3000.0 mS/cm² | Transient Na (Nav1.6) |
| `g_nap` | 5.0 mS/cm² | Persistent Na (Nav1.6) |
| `g_ks` | 80.0 mS/cm² | Slow K (Kv7/KCNQ) |
| `g_l` | 7.0 mS/cm² | Nodal leak |
| `e_na` | 50.0 mV | Na reversal |
| `e_k` | -90.0 mV | K reversal |
| `e_l` | -90.0 mV | Leak reversal |
| `sub_steps` | 20 | dt_sub = 0.025 ms |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/misc.rs` |
| PyO3 wrapper | Yes (state: v, m, h, p, s) |
| NetworkRunner wired | `NeuronVariant::NodeOfRanvier` |
| `create_neuron("NodeOfRanvier")` | Yes |
| `supported_models()` | Includes "NodeOfRanvier" |
| STRONG tests | 10 |
| Benchmark | `node_of_ranvier_1k_steps`: **3.99 ms** (3.99 µs/step), i5-11600K |
