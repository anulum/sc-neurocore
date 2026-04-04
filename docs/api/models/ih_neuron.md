# IhNeuron

**Module:** `engine/src/neurons/channels.rs`
**Reference:** Robinson & Bhatt, Neuron 11:953, 1993; Pape, Annu Rev Physiol 58:299, 1996
**Family:** WB Na+/K+ base + Ih (HCN, hyperpolarisation-activated cation current)
**State variables:** `v`, `h` (Na+ inactivation), `n` (Kdr), `r` (Ih activation)

---

## Biological Context

The hyperpolarisation-activated cation current (Ih, carried by HCN channels) is unique among voltage-gated channels: it activates upon hyperpolarisation rather than depolarisation. It conducts a mixed Na+/K+ current with reversal potential ~-40 mV, making it depolarising at typical resting potentials.

Key features:
- **Voltage sag**: during sustained hyperpolarisation, Ih gradually activates and depolarises the membrane back towards rest, producing the characteristic "sag" in voltage recordings
- **Rebound excitation**: Ih that accumulates during inhibition persists briefly after release, depolarising the cell past rest and facilitating spike generation
- **Pacemaker oscillations**: in thalamic relay neurons and cardiac SA node, Ih interacts with T-type Ca2+ to produce rhythmic oscillations
- **Temporal integration**: Ih normalises the time course of synaptic integration across dendritic compartments

---

## Equations

### WB base + Ih

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_h - I_L + I_{ext}$$

$$I_h = g_h \cdot r \cdot (V - E_h)$$

$$r_\infty = \frac{1}{1 + \exp((V + 80)/10)}$$

$$\tau_r = 100 + \frac{200}{1 + \exp((V + 70)/10)}$$

---

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `h` | 0.6 | — | Transient Na+ inactivation |
| `n` | 0.32 | — | Kdr activation |
| `r` | 0.1 | — | Ih activation |
| `g_na` | 35.0 | mS/cm² | Transient Na+ |
| `g_k` | 9.0 | mS/cm² | Kdr |
| `g_h` | 0.15 | mS/cm² | Ih conductance |
| `g_l` | 0.2 | mS/cm² | Leak |
| `e_na` | 55.0 | mV | Na+ reversal |
| `e_k` | -90.0 | mV | K+ reversal |
| `e_h` | -40.0 | mV | Ih reversal (mixed cation) |
| `e_l` | -65.0 | mV | Leak reversal |
| `phi` | 5.0 | — | Kinetic scaling |
| `dt` | 0.5 | ms | Timestep (50 sub-steps) |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/channels.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, h, n, r) |
| NetworkRunner wired | `NeuronVariant::Ih` |
| `create_neuron("Ih")` | Yes |
| `supported_models()` | Includes "Ih" |
| STRONG tests | 11 (fire, silent, sag potential, r-gate activation, rebound, negative, NaN, extreme, reset, gates, performance) |
| Benchmark | `ih_1k_steps`: **5.17 ms** (5.17 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| ih_1k_steps | 5.17 ms |
| Per step | **5.17 µs** |

WB gating with 50 sub-steps + Ih. Measured 2026-04-04.

---

## Findings

1. **Fires with excitatory input.** Sustained spiking with I=2. Verified.
2. **Silent without input.** No spontaneous firing at rest. Verified.
3. **Sag potential.** Ih depolarises membrane during sustained hyperpolarisation. Verified.
4. **r gate activates on hyperpolarisation.** r increases during negative input. Verified.
5. **Rebound excitation.** After hyperpolarisation, accumulated Ih facilitates spike generation. Verified.
6. **Reset clears state.** All variables return to initial values. Verified.
