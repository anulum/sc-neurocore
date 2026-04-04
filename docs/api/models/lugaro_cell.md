# LugaroCell

**Module:** `engine/src/neurons/cerebellar.rs`
**Reference:** Dieudonné & Bhatt, J Physiol 548:97, 2003; Lainé & Bhatt, Front Syst Neurosci 1:4, 2007
**Family:** LIF + adaptation + serotonin modulation
**State variables:** `v` (membrane potential), `adapt` (adaptation current)

---

## Biological Context

Lugaro cells are rare fusiform interneurons in the cerebellar granular layer, estimated at ~1% of granular layer neurons. They have large horizontally oriented somata with long axonal projections that inhibit Golgi cells and molecular layer interneurons (stellate, basket cells).

Key features:
- **Serotonin sensitivity**: 5-HT from brainstem raphe nuclei increases Lugaro cell excitability, enhancing their inhibition of Golgi cells
- **Adaptation**: moderate spike frequency adaptation limits sustained firing
- **Disinhibitory circuit**: by inhibiting Golgi cells, Lugaro cells can disinhibit granule cells
- **Regular spiking**: fires at moderate rates (5-15 Hz) with excitatory input

---

## Equations

### LIF with adaptation

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) - adapt + g_{eff} \cdot I_{ext}$$

$$\tau_{adapt} \frac{d(adapt)}{dt} = a_{adapt}(V - V_{rest}) - adapt$$

On spike: $adapt \leftarrow adapt + 1.0$

### Serotonin modulation

$$g_{eff} = gain \cdot (1 + 0.5 \cdot [5\text{-}HT])$$

---

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `v` | -55.0 | mV | Membrane potential |
| `adapt` | 0.0 | mV | Adaptation current |
| `v_rest` | -55.0 | mV | Resting potential |
| `v_reset` | -65.0 | mV | Post-spike reset |
| `v_threshold` | -48.0 | mV | Spike threshold |
| `tau_m` | 10.0 | ms | Membrane time constant |
| `tau_adapt` | 150.0 | ms | Adaptation time constant |
| `a_adapt` | 0.05 | — | Adaptation coupling |
| `gain` | 2.0 | — | Input scaling |
| `serotonin` | 0.0 | — | 5-HT level [0, 1] |
| `dt` | 0.5 | ms | Integration timestep |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/cerebellar.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, adapt) |
| NetworkRunner wired | `NeuronVariant::Lugaro` |
| `create_neuron("LugaroCell")` | Yes |
| `supported_models()` | Includes "LugaroCell" |
| STRONG tests | 10 (fire, low-threshold, adaptation, 5-HT, adapt increase, negative, NaN, extreme, reset, performance) |
| Pipeline integration | Covered by `create_neuron_all_supported` |
| Benchmark | `lugaro_10k_steps`: **164 µs** (16.4 ns/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| lugaro_10k_steps | 164 µs |
| Per step | **16.4 ns** |

Simple LIF + adaptation, no sub-stepping. Measured 2026-04-04.

---

## Findings

1. **Fires with excitatory input.** Sustained spiking with I=5. Verified.
2. **Fires easily with moderate input.** Low effective threshold from depolarised rest. Verified.
3. **Adaptation slows firing.** Early epochs fire more than late epochs. Verified.
4. **Serotonin increases firing.** 5-HT=1.0 produces more spikes than 5-HT=0. Verified.
5. **Adaptation current increases during spiking.** adapt > 0 after sustained firing. Verified.
6. **Reset clears state.** All variables return to initial values. Verified.
