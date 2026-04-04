# UnipolarBrushCell

**Module:** `engine/src/neurons/cerebellar.rs`
**Reference:** Bhatt et al., J Comp Neurol 349:560, 1994; Diana et al., J Neurosci 27:4374, 2007
**Family:** LIF + slow persistent (NMDA-like) current
**State variables:** `v` (membrane potential), `persistent` (slow persistent current)

---

## Biological Context

Unipolar brush cells (UBCs) are excitatory glutamatergic interneurons unique to the granular layer of the vestibulocerebellum (flocculus, nodulus, uvula). They receive mossy fibre input via a single large brush-like dendrite that forms a giant synapse, creating a 1:1 relay with signal amplification and prolongation.

Key features:
- **Persistent activity**: slow NMDA-like current sustains depolarisation after input ceases
- **Signal amplification**: transforms brief mossy fibre bursts into prolonged granule cell activation
- **Vestibular processing**: critical for timing in vestibulo-ocular reflex circuits
- **Excitatory interneuron**: one of the few excitatory interneurons in the cerebellum

---

## Equations

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + g \cdot \max(I_{ext}, 0) + I_{persistent}$$

$$\tau_p \frac{dI_p}{dt} = p_{gain} \cdot g \cdot \max(I_{ext}, 0) - I_p$$

---

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `persistent` | 0.0 | mV | Persistent current |
| `v_rest` | -65.0 | mV | Resting potential |
| `v_reset` | -70.0 | mV | Post-spike reset |
| `v_threshold` | -50.0 | mV | Spike threshold |
| `tau_m` | 8.0 | ms | Membrane time constant |
| `tau_persistent` | 200.0 | ms | Persistent current decay |
| `persistent_gain` | 0.5 | — | Input→persistent coupling |
| `gain` | 2.5 | — | Input scaling |
| `dt` | 0.5 | ms | Integration timestep |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/cerebellar.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, persistent) |
| NetworkRunner wired | `NeuronVariant::UnipolarBrush` |
| `create_neuron("UnipolarBrushCell")` | Yes |
| `supported_models()` | Includes "UnipolarBrushCell" |
| STRONG tests | 9 (fire, silent, persistent activity, post-input, negative, NaN, extreme, reset, performance) |
| Benchmark | `ubc_10k_steps`: **93 µs** (9.3 ns/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| ubc_10k_steps | 93 µs |
| Per step | **9.3 ns** |

Simple LIF + persistent current, no sub-stepping. Measured 2026-04-04.

---

## Findings

1. **Fires with excitatory input.** Sustained spiking with I=5. Verified.
2. **Silent without input.** No spontaneous firing. Verified.
3. **Persistent current builds during input.** persistent > 0 after sustained drive. Verified.
4. **Persistent current decays after input removal.** Slow exponential decay. Verified.
5. **Reset clears state.** v=-65, persistent=0 after reset. Verified.
