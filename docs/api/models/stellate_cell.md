# StellateCell

**Module:** `engine/src/neurons/cerebellar.rs`
**Reference:** Sultan & Bower, J Comp Neurol 409:63, 1999; Häusser & Clark, Neuron 19:665, 1997
**Family:** WB Na+/K+ core + Kv3.1 (fast-spiking)
**State variables:** `v`, `h` (Na+ inactivation), `n` (Kdr), `p` (Kv3.1 activation)

---

## Biological Context

Cerebellar stellate cells are small inhibitory interneurons in the molecular layer. They receive excitatory input from parallel fibres (granule cell axons) and provide feedforward GABAergic inhibition onto Purkinje cell dendrites. Stellate cells innervate more distal dendritic regions than basket cells.

Key features:
- **Fast-spiking**: narrow action potentials, minimal adaptation, sustained high-frequency firing
- **Kv3.1 channels**: enable rapid repolarisation for high-frequency firing (similar to PV+ interneurons)
- **Feedforward inhibition**: shapes Purkinje cell dendritic integration and timing
- **Small soma**: higher input resistance than basket cells

---

## Equations

### WB gating with phi scaling + Kv3.1

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_{Kv3} - I_L + I_{ext}$$

$$I_{Na} = g_{Na} m_\infty^3 h (V - E_{Na})$$
$$I_K = g_K n^4 (V - E_K)$$
$$I_{Kv3} = g_{Kv3} p^2 (V - E_K)$$

Gate kinetics: WB alpha/beta rates with phi=5 (via `safe_rate`), m uses steady-state approximation.

### Kv3.1 gating

$$p_\infty = \frac{1}{1 + \exp(-(V+10)/10)}$$
$$\tau_p = 1 + \frac{4}{1 + \exp((V+20)/15)}$$

---

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `h` | 0.6 | — | Na+ inactivation |
| `n` | 0.32 | — | Kdr activation |
| `p` | 0.0 | — | Kv3.1 activation |
| `g_na` | 35.0 | mS/cm² | Na+ conductance |
| `g_k` | 9.0 | mS/cm² | Kdr conductance |
| `g_kv3` | 3.0 | mS/cm² | Kv3.1 conductance |
| `g_l` | 0.1 | mS/cm² | Leak conductance |
| `e_na` | 55.0 | mV | Na+ reversal |
| `e_k` | -90.0 | mV | K+ reversal |
| `e_l` | -65.0 | mV | Leak reversal |
| `c_m` | 0.5 | µF/cm² | Specific capacitance (small soma) |
| `phi` | 5.0 | — | Kinetic scaling factor |
| `dt` | 0.5 | ms | Integration timestep (50 sub-steps) |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/cerebellar.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, h, n, p) |
| NetworkRunner wired | `NeuronVariant::Stellate` |
| `create_neuron("StellateCell")` | Yes |
| `supported_models()` | Includes "StellateCell" |
| STRONG tests | 11 (fire, silent, high-freq, minimal adaptation, Kv3.1, negative, NaN, extreme, reset, gates, performance) |
| Pipeline integration | Covered by `create_neuron_all_supported` |
| Benchmark | `stellate_1k_steps`: **5.58 ms** (5.58 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| stellate_1k_steps | 5.58 ms |
| Per step | **5.58 µs** |

WB gating with 50 sub-steps + Kv3.1. Measured 2026-04-04.

---

## Findings

1. **Fires with excitatory input.** Sustained spiking with I=2. Verified.
2. **Silent without input.** No spontaneous firing at rest. Verified.
3. **High-frequency firing.** >100 Hz with strong drive. Verified.
4. **Minimal adaptation.** Early and late spike counts similar. Verified.
5. **Kv3.1 enables narrow spikes.** Model fires with Kv3.1 present. Verified.
6. **Reset clears state.** All variables return to initial values. Verified.
