# DCNNeuron

**Module:** `engine/src/neurons/cerebellar.rs`
**Reference:** Llinás & Mühlethaler, J Physiol 404:241, 1988; Jahnsen, J Physiol 372:129, 1986
**Family:** WB Na+/K+ + T-type Ca2+ + Ih (rebound bursting)
**State variables:** `v`, `h`, `n` (WB gating), `s` (T-type inactivation), `r` (Ih activation)

---

## Biological Context

Deep cerebellar nuclei (DCN) neurons are the main output neurons of the cerebellum. They receive massive GABAergic inhibition from Purkinje cells and excitatory input from mossy fibre and climbing fibre collaterals. DCN neurons relay cerebellar computations to the thalamus, brainstem, and spinal cord.

Key features:
- **Rebound bursting**: when Purkinje cell inhibition pauses, T-type Ca2+ channels that de-inactivated during hyperpolarisation produce a burst of spikes — the primary cerebellar timing mechanism
- **Ih (HCN) current**: hyperpolarisation-activated mixed cation current provides a depolarising "sag" and contributes to pacemaker properties
- **High spontaneous rate**: in vivo, DCN neurons fire at 30-100 Hz, modulated by Purkinje cell inhibition
- **Sole cerebellar output**: all cerebellar computations must pass through ~50,000 DCN neurons per nucleus

---

## Equations

### WB Na+/K+ gating + T-type + Ih

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_T - I_h - I_L + I_{ext}$$

$$I_T = g_T m_{T,\infty}^2 s (V - E_{Ca})$$
$$I_h = g_h r (V - E_h)$$

Gate kinetics: WB alpha/beta with phi=5 (via `safe_rate`), m uses steady-state.

### T-type Ca2+ and Ih gating

Same as GranuleCell T-type: $m_{T,\infty}$, $s_\infty$, $\tau_s$.

$$r_\infty = \frac{1}{1 + \exp((V+80)/10)}$$
$$\tau_r = 100 + \frac{200}{1 + \exp((V+70)/10)}$$

---

## Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `v` | -60.0 | mV | Membrane potential |
| `h` | 0.6 | — | Na+ inactivation |
| `n` | 0.32 | — | Kdr activation |
| `s` | 0.8 | — | T-type inactivation |
| `r` | 0.1 | — | Ih activation |
| `g_na` | 35.0 | mS/cm² | Na+ conductance |
| `g_k` | 9.0 | mS/cm² | Kdr conductance |
| `g_t` | 0.1 | mS/cm² | T-type Ca2+ |
| `g_h` | 0.02 | mS/cm² | Ih conductance |
| `g_l` | 0.2 | mS/cm² | Leak conductance |
| `e_na` | 55.0 | mV | Na+ reversal |
| `e_k` | -90.0 | mV | K+ reversal |
| `e_ca` | 120.0 | mV | Ca2+ reversal |
| `e_h` | -40.0 | mV | Ih reversal (mixed cation) |
| `e_l` | -65.0 | mV | Leak reversal |
| `phi` | 5.0 | — | Kinetic scaling |
| `dt` | 0.5 | ms | Timestep (50 sub-steps) |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/cerebellar.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, h, n, s, r) |
| NetworkRunner wired | `NeuronVariant::DCN` |
| `create_neuron("DCNNeuron")` | Yes |
| `supported_models()` | Includes "DCNNeuron" |
| STRONG tests | 10 (fire, silent, rebound, Ih depolarisation, negative, NaN, extreme, reset, gates, performance) |
| Benchmark | `dcn_1k_steps`: **4.83 ms** (4.83 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| dcn_1k_steps | 4.83 ms |
| Per step | **4.83 µs** |

WB gating with 50 sub-steps + T-type Ca2+ + Ih. Measured 2026-04-04.

---

## Findings

1. **Fires with excitatory input.** Sustained spiking with I=5. Verified.
2. **Silent without input.** No spontaneous firing at rest. Verified.
3. **Rebound burst via T-type.** De-inactivated T-type facilitates firing after hyperpolarisation. Verified.
4. **Ih depolarises from hyperpolarised state.** With Ih, resting potential is more depolarised than without. Verified.
5. **Reset clears state.** All variables return to initial values. Verified.
6. **All gates bounded [0, 1].** After extensive simulation. Verified.
