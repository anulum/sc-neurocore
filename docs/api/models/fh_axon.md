# FrankenhaeUserHuxleyAxon

**Module:** `engine/src/neurons/misc.rs`
**Reference:** Frankenhaeuser & Huxley, J Physiol 171:302, 1964
**Family:** Myelinated nerve fibre (permeability-based HH variant)
**State variables:** `v` (membrane potential), `m` (Na activation), `h` (Na inactivation), `n` (K delayed rectifier), `p` (slow non-specific)

---

## Biological Context

The Frankenhaeuser-Huxley model describes action potential propagation at nodes of Ranvier in myelinated nerve fibres (originally Xenopus sciatic nerve). It differs from the classic Hodgkin-Huxley model:

- **Permeability-based**: uses P_Na, P_K, P_p instead of conductances
- **4 gating variables**: m, h, n, plus p (slow non-specific current)
- **Smaller AP amplitude**: myelinated nodes have different channel densities
- **Different kinetics**: rate constants tuned for frog myelinated nerve at 20°C

Key features:
- **4 gating variables**: m² (Na), h (Na inact), n² (K), p² (slow)
- **50 sub-steps**: dt_sub = 0.01 ms for numerical stability
- **Permeability formulation**: closer to GHK than conductance-based HH

---

## Equations

$$C_m \frac{dV}{dt} = -(I_{Na} + I_K + I_p + I_L) + I_{ext}$$

$$I_{Na} = P_{Na} \cdot m^2 \cdot h \cdot (V - E_{Na})$$
$$I_K = P_K \cdot n^2 \cdot (V - E_K)$$
$$I_p = P_p \cdot p^2 \cdot (V - E_p)$$

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `c_m` | 2.0 µF/cm² | Node capacitance |
| `p_na` | 12.0 | Na permeability |
| `p_k` | 1.2 | K permeability |
| `p_p` | 0.54 | Slow current permeability |
| `g_l` | 0.3 | Leak conductance |
| `e_na` | 115.0 mV | Na reversal (above rest) |
| `e_k` | -12.0 mV | K reversal |
| `sub_steps` | 50 | Sub-steps per external step |
| `dt` | 0.5 ms | External step |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/misc.rs` |
| PyO3 wrapper | Yes (state: v, m, h, n, p) |
| NetworkRunner wired | `NeuronVariant::FHAxon` |
| `create_neuron("FrankenhaeUserHuxleyAxon")` | Yes |
| `supported_models()` | Includes "FrankenhaeUserHuxleyAxon" |
| STRONG tests | 10 |
| Benchmark | `fh_axon_1k_steps`: **19.88 ms** (19.88 µs/step), i5-11600K |
