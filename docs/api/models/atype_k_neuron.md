# ATypeKNeuron

**Module:** `engine/src/neurons/channels.rs`
**Reference:** Connor & Stevens, J Physiol 213:31, 1971; Hoffman et al., Nature 387:869, 1997
**Family:** WB Na+/K+ base + A-type K+ (IA, transient outward)
**State variables:** `v`, `h` (Na+ inactivation), `n` (Kdr), `a` (IA activation), `b` (IA inactivation)

---

## Biological Context

The A-type potassium current (IA) is a transient outward K+ current that activates rapidly at subthreshold voltages and inactivates over tens of milliseconds. IA opposes depolarisation, creating a characteristic delay before the first action potential and controlling interspike intervals.

Key features:
- **First-spike latency**: IA must inactivate before sufficient depolarisation for a spike
- **Spike frequency control**: IA recovery during interspike intervals limits firing rate
- **Coincidence detection**: neurons with strong IA preferentially respond to synchronous inputs
- **Dendritic processing**: IA in hippocampal CA1 dendrites regulates back-propagating APs

---

## Equations

### WB base + IA

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_A - I_L + I_{ext}$$

$$I_A = g_A \cdot a^3 \cdot b \cdot (V - E_K)$$

$$a_\infty = \frac{1}{1 + \exp(-(V+50)/20)}, \quad \tau_a = 2 \text{ ms}$$
$$b_\infty = \frac{1}{1 + \exp((V+70)/6)}, \quad \tau_b = 50 \text{ ms}$$

---

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `a` | 0.1 | — | IA activation |
| `b` | 0.8 | — | IA inactivation |
| `g_a` | 8.0 | mS/cm² | A-type K+ conductance |
| `phi` | 5.0 | — | Kinetic scaling |
| `dt` | 0.5 | ms | Timestep (50 sub-steps) |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/channels.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, h, n, a, b) |
| NetworkRunner wired | `NeuronVariant::ATypeK` |
| `create_neuron("ATypeK")` | Yes |
| `supported_models()` | Includes "ATypeK" |
| STRONG tests | 10 (fire, silent, delay, rate reduction, negative, NaN, extreme, reset, gates, performance) |
| Benchmark | `atype_k_1k_steps`: **4.00 ms** (4.00 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| atype_k_1k_steps | 4.00 ms |
| Per step | **4.00 µs** |

WB gating with 50 sub-steps + IA. Measured 2026-04-04.

---

## Findings

1. **Fires with excitatory input.** Sustained spiking with I=3. Verified.
2. **Silent without input.** No spontaneous firing. Verified.
3. **IA delays first spike.** Removing g_a shortens latency to first spike. Verified.
4. **IA reduces firing rate.** More spikes without IA at same input. Verified.
5. **Reset clears state.** a=0.1, b=0.8 after reset. Verified.
