# PersistentNaNeuron

**Module:** `engine/src/neurons/channels.rs`
**Reference:** Crill, Annu Rev Physiol 58:349, 1996; French et al., Neuroscience 42:363, 1990
**Family:** WB Na+/K+ base + persistent Na+ current (INaP)
**State variables:** `v`, `h` (Na+ inactivation), `n` (Kdr), `p` (INaP activation)

---

## Biological Context

The persistent sodium current (INaP) is a non-inactivating Na+ current that activates at subthreshold voltages (-60 to -40 mV). Unlike transient Na+ (which inactivates within milliseconds), INaP remains active as long as the membrane is depolarised, providing sustained depolarising drive.

Key features:
- **Subthreshold amplification**: INaP amplifies synaptic inputs near threshold, increasing neuronal sensitivity
- **Subthreshold oscillations**: in entorhinal cortex layer II stellate cells, INaP drives theta-frequency oscillations
- **Plateau potentials**: in spinal motoneurons, INaP enables bistable membrane states (rest vs sustained firing)
- **Burst generation**: in pre-Bötzinger complex respiratory neurons, INaP drives rhythmic bursting
- **Spontaneous activity**: neurons with strong INaP may fire spontaneously

---

## Equations

### WB base + INaP

$$C_m \frac{dV}{dt} = -I_{Na} - I_{NaP} - I_K - I_L + I_{ext}$$

$$I_{NaP} = g_{NaP} \cdot p \cdot (V - E_{Na})$$

$$p_\infty = \frac{1}{1 + \exp(-(V + 48)/5)}$$

$$\tau_p = 10 + \frac{40}{1 + ((V + 48)/10)^2}$$

Gate kinetics: WB alpha/beta with phi=5 (via `safe_rate`), transient Na+ m uses steady-state.

---

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `h` | 0.6 | — | Transient Na+ inactivation |
| `n` | 0.32 | — | Kdr activation |
| `p` | 0.0 | — | INaP activation |
| `g_na` | 35.0 | mS/cm² | Transient Na+ |
| `g_nap` | 0.15 | mS/cm² | Persistent Na+ |
| `g_k` | 9.0 | mS/cm² | Kdr |
| `g_l` | 0.3 | mS/cm² | Leak (higher to counteract INaP) |
| `e_na` | 55.0 | mV | Na+ reversal |
| `e_k` | -90.0 | mV | K+ reversal |
| `e_l` | -65.0 | mV | Leak reversal |
| `phi` | 5.0 | — | Kinetic scaling |
| `dt` | 0.5 | ms | Timestep (50 sub-steps) |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/channels.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, h, n, p) |
| NetworkRunner wired | `NeuronVariant::PersistentNa` |
| `create_neuron("PersistentNa")` | Yes |
| `supported_models()` | Includes "PersistentNa" |
| STRONG tests | 11 (fire, subthreshold oscillations, lower threshold, p-gate activation, rate increase, negative, NaN, extreme, reset, gates, performance) |
| Benchmark | `persistent_na_1k_steps`: **3.06 ms** (3.06 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| persistent_na_1k_steps | 3.06 ms |
| Per step | **3.06 µs** |

WB gating with 50 sub-steps + INaP. Measured 2026-04-04.

---

## Findings

1. **Fires with excitatory input.** Sustained spiking with I=2. Verified.
2. **INaP drives subthreshold activity.** With inhibitory input, neuron is silenced. Verified.
3. **INaP lowers effective threshold.** More spikes with INaP than without at same input. Verified.
4. **p gate activates at subthreshold voltages.** p > 0 after simulation near -50 mV. Verified.
5. **Higher g_nap increases firing rate.** Dose-response relationship confirmed. Verified.
6. **Reset clears state.** All variables return to initial values. Verified.
