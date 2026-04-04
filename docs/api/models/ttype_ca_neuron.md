# TTypeCaNeuron

**Module:** `engine/src/neurons/channels.rs`
**Reference:** Huguenard, Annu Rev Physiol 58:329, 1996; Destexhe et al., J Neurophysiol 76:2049, 1996
**Family:** WB Na+/K+ base + T-type Ca2+ (IT, low-voltage-activated)
**State variables:** `v`, `h` (Na+ inactivation), `n` (Kdr), `s` (T-type inactivation)

---

## Biological Context

T-type (transient) Ca2+ channels are low-voltage-activated (LVA) channels that open at subthreshold membrane potentials and inactivate with sustained depolarisation. The key feature is voltage-dependent de-inactivation: prolonged hyperpolarisation removes inactivation, priming the channels for a large transient Ca2+ influx upon depolarisation.

Key features:
- **Low-threshold spikes (LTS)**: broad Ca2+ depolarisation that can trigger bursts of Na+ APs
- **Rebound bursting**: after inhibition, de-inactivated T-type channels produce LTS + Na+ burst
- **Sleep spindles**: T-type in thalamic reticular nucleus drives 7-14 Hz oscillations
- **Spike inactivation**: each spike strongly inactivates T-type, limiting burst duration

---

## Equations

### WB base + T-type Ca2+

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_T - I_L + I_{ext}$$

$$I_T = g_T \cdot m_{T,\infty}^2 \cdot s \cdot (V - E_{Ca})$$

$$m_{T,\infty} = \frac{1}{1 + \exp(-(V + 52)/5)}$$

$$s_\infty = \frac{1}{1 + \exp((V + 81)/4)}$$

$$\tau_s = 30 + \frac{100}{1 + \exp((V + 75)/10)}$$

On spike: $s \leftarrow 0.3 \cdot s$ (strong inactivation).

---

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `h` | 0.6 | — | Na+ inactivation |
| `n` | 0.32 | — | Kdr activation |
| `s` | 0.9 | — | T-type inactivation |
| `g_na` | 35.0 | mS/cm² | Transient Na+ |
| `g_k` | 9.0 | mS/cm² | Kdr |
| `g_t` | 0.1 | mS/cm² | T-type Ca2+ |
| `g_l` | 0.2 | mS/cm² | Leak |
| `e_ca` | 120.0 | mV | Ca2+ reversal |
| `phi` | 5.0 | — | Kinetic scaling |
| `dt` | 0.5 | ms | Timestep (50 sub-steps) |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/channels.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, h, n, s) |
| NetworkRunner wired | `NeuronVariant::TTypeCa` |
| `create_neuron("TTypeCa")` | Yes |
| `supported_models()` | Includes "TTypeCa" |
| STRONG tests | 11 (fire, silent, rebound, s de-inactivation, spike inactivation, negative, NaN, extreme, reset, gates, performance) |
| Benchmark | `ttype_ca_1k_steps`: **3.94 ms** (3.94 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| ttype_ca_1k_steps | 3.94 ms |
| Per step | **3.94 µs** |

WB gating with 50 sub-steps + T-type Ca2+. Measured 2026-04-04.

---

## Findings

1. **Fires with excitatory input.** Sustained spiking with I=2. Verified.
2. **Silent without input.** No spontaneous firing at rest. Verified.
3. **Rebound burst.** De-inactivated T-type facilitates firing after hyperpolarisation. Verified.
4. **s gate de-inactivates at hyperpolarised potentials.** s increases during negative input. Verified.
5. **Spike inactivates T-type.** s decreases after spike (s *= 0.3). Verified.
6. **Reset clears state.** v=-65, s=0.9 after reset. Verified.
