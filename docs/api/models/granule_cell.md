# GranuleCell

**Module:** `engine/src/neurons/cerebellar.rs`
**Reference:** D'Angelo et al., J Neurosci 21(3), 2001; Bhalla & Bhatt, Cerebellum, 2012
**Family:** LIF + tonic GABA inhibition + T-type Ca2+ rebound
**State variables:** `v` (membrane potential), `s` (T-type Ca2+ inactivation)

---

## Biological Context

Cerebellar granule cells are the most numerous neurons in the brain, comprising approximately 50% of all neurons (~50 billion in humans). They have tiny somata (6-8 µm diameter), resulting in very high input resistance and short membrane time constants. Each granule cell receives input from 4 mossy fibres via specialised structures called glomeruli, where Golgi cell axons provide tonic GABAergic inhibition.

Key features:
- **Tonic GABA inhibition:** Golgi cells provide continuous shunting inhibition via extrasynaptic GABA_A receptors, keeping granule cells near but below threshold. This creates a high-pass filter for mossy fibre input.
- **T-type Ca2+ channels:** Enable post-inhibitory rebound bursting. When tonic inhibition is released (disinhibition), T-type channels that de-inactivated during hyperpolarisation produce a transient depolarising current, triggering brief bursts.
- **High input resistance:** Small soma means even modest synaptic currents produce large voltage changes.
- **Parallel fibre output:** Granule cell axons ascend and bifurcate into parallel fibres that extend for several millimetres along the folium, synapsing onto Purkinje cell dendrites.
- **Sparse coding:** At any given time, only ~1-3% of granule cells are active, providing a combinatorial expansion of mossy fibre representations.

---

## Equations

### Membrane dynamics (LIF with tonic inhibition and T-type Ca2+)

$$\tau_m \frac{dV}{dt} = -g_l(V - E_l) - g_{tonic}(V - E_{GABA}) - I_T + g \cdot \max(I_{ext}, 0)$$

### T-type Ca2+ current

$$I_T = g_T \cdot m_{T,\infty}^2 \cdot s \cdot (V - E_{Ca})$$

$$m_{T,\infty} = \frac{1}{1 + \exp(-(V + 52)/5)}$$

$$\tau_s \frac{ds}{dt} = s_\infty - s$$

$$s_\infty = \frac{1}{1 + \exp((V + 60)/6.5)}$$

$$\tau_s = 20 + \frac{50}{1 + \exp((V + 65)/10)}$$

### Spike rule

$$V \geq V_{thresh} \Rightarrow V \leftarrow V_{reset}, \quad s \leftarrow 0.5 \cdot s$$

---

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `v` | -70.0 | mV | Membrane potential |
| `s` | 0.95 | — | T-type Ca2+ inactivation |
| `g_l` | 0.05 | mS/cm² | Leak conductance (low for high Rin) |
| `g_tonic` | 0.02 | mS/cm² | Tonic GABA conductance |
| `g_t` | 0.03 | mS/cm² | T-type Ca2+ conductance |
| `e_l` | -70.0 | mV | Leak reversal |
| `e_gaba` | -75.0 | mV | GABA reversal (shunting) |
| `e_ca` | 120.0 | mV | Ca2+ reversal |
| `tau_m` | 5.0 | ms | Membrane time constant |
| `c_m` | 1.0 | µF/cm² | Specific capacitance |
| `v_threshold` | -40.0 | mV | Spike threshold |
| `v_reset` | -70.0 | mV | Post-spike reset |
| `refrac_period` | 1.0 | ms | Refractory period |
| `gain` | 1.5 | — | Input scaling |
| `dt` | 0.5 | ms | Integration timestep |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/cerebellar.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, s) |
| NetworkRunner wired | `NeuronVariant::Granule` |
| `create_neuron("GranuleCell")` | Yes |
| `supported_models()` | Includes "GranuleCell" |
| STRONG tests | 11 (fire, no-fire, silent, GABA threshold, rebound, negative, NaN, extreme, reset, high Rin, performance) |
| Pipeline integration | Covered by `create_neuron_all_supported` |
| Benchmark | `granule_10k_steps`: **466 µs** (46.6 ns/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| granule_10k_steps | 466 µs |
| Per step | **46.6 ns** |

Simple Euler integration with T-type Ca2+ gating, no sub-stepping. Measured 2026-04-04.

---

## Findings

1. **Silent at rest.** Tonic GABA inhibition keeps the cell below threshold without external input. Verified.
2. **Fires with strong input.** Sufficient excitatory current overcomes tonic inhibition. Verified.
3. **Tonic GABA raises effective threshold.** Removing g_tonic increases firing rate for same input. Verified.
4. **Rebound burst via T-type.** De-inactivated T-type channels (high s) facilitate firing after hyperpolarisation. Verified.
5. **High input resistance.** Small current produces large voltage deflection. Verified.
6. **Reset clears state.** v=-70, s=0.95 after reset. Verified.
