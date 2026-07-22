# MartinottiNeuron

**Module:** `engine/src/neurons/interneurons/martinotti_neuron.rs`
**Reference:** Silberberg & Markram, J. Physiol. 580(1), 2007 / Toledo-Rodriguez et al., Cereb. Cortex 15(7), 2005
**Family:** Hodgkin-Huxley variant, adapting GABAergic interneuron
**State variables:** `v` (membrane potential), `m` (Na+ activation), `h` (Na+ inactivation), `n` (K+ activation), `p` (M-current activation), `s` (T-type Ca2+ inactivation)

---

## Biological Context

Martinotti cells are adapting interneurons with ascending axons that arborise in layer 1, targeting apical dendrites of pyramidal neurons across multiple cortical columns. They are a morphological subtype within the SST+ population, distinguished by stronger adaptation and lower rheobase.

Key electrophysiological features:
- Pronounced spike-frequency adaptation (strong M-current, Kv7)
- Low rheobase due to reduced capacitance (C_m=0.8)
- T-type Ca2+ for rebound spiking
- Facilitating synaptic input from L5 pyramidal cells
- Ascending axon targeting layer 1 (distal apical dendrites)
- Lower sustained firing rates than SST+ (~15-40 Hz)

The very strong M-current (g_m=0.25, more than double SST's 0.12) makes adaptation the defining feature. At constant input, firing rate drops substantially within the first 200-500 ms before reaching a low steady state.

---

## Equations

### Core currents (Na+, K+, M-current, T-type Ca2+, leak)

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_M - I_T - I_L + I_{ext}$$

$$I_{Na} = g_{Na} \, m^3 \, h \, (V - E_{Na})$$

$$I_K = g_K \, n^4 \, (V - E_K)$$

### Na+ gating (Pospischil alpha/beta, $V_T = -56.2$ mV)

$$\alpha_m = \frac{-0.32 (V - V_T - 13)}{e^{-(V - V_T - 13)/4} - 1}, \quad \beta_m = \frac{0.28 (V - V_T - 17)}{e^{(V - V_T - 17)/5} - 1}$$

$$\alpha_h = 0.128 \, e^{-(V - V_T - 17)/18}, \quad \beta_h = \frac{4}{1 + e^{-(V - V_T - 40)/5}}$$

### K+ gating

$$\alpha_n = \frac{-0.032 (V - V_T - 15)}{e^{-(V - V_T - 15)/5} - 1}, \quad \beta_n = 0.5 \, e^{-(V - V_T - 10)/40}$$

### M-current (Kv7, very strong for Martinotti)

$$p_\infty = \frac{1}{1 + e^{-(V + 35)/10}}$$

$$\tau_p = \frac{400}{3.3 \, e^{(V + 35)/20} + e^{-(V + 35)/20}}$$

$$I_M = g_M \, p \, (V - E_K)$$

### T-type Ca2+ (low-threshold burst)

$$m_{T,\infty} = \frac{1}{1 + e^{-(V + 57)/6.2}}$$

$$s_\infty = \frac{1}{1 + e^{(V + 81)/4}}, \quad \tau_s = 30 + \frac{200}{1 + e^{(V + 70)/5}}$$

$$I_T = g_T \, m_{T,\infty}^2 \, s \, (V - E_{Ca})$$

Sub-stepping: 4 steps per call (0.1 ms real time per call at dt=0.025).

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `m` | 0.02 | -- | Na+ activation gate |
| `h` | 0.8 | -- | Na+ inactivation gate |
| `n` | 0.2 | -- | Delayed-rectifier K+ activation |
| `p` | 0.0 | -- | M-current (Kv7) activation |
| `s` | 0.9 | -- | T-type Ca2+ inactivation |
| `g_na` | 40.0 | mS/cm^2 | Na+ conductance |
| `g_k` | 5.0 | mS/cm^2 | Delayed-rectifier K+ |
| `g_m` | 0.25 | mS/cm^2 | M-current (very strong adaptation) |
| `g_t` | 0.01 | mS/cm^2 | T-type Ca2+ (minimal window current) |
| `g_l` | 0.05 | mS/cm^2 | Leak conductance |
| `e_na` | 50.0 | mV | Na+ reversal |
| `e_k` | -90.0 | mV | K+ reversal |
| `e_ca` | 120.0 | mV | Ca2+ reversal |
| `e_l` | -65.0 | mV | Leak reversal |
| `c_m` | 0.8 | uF/cm^2 | Membrane capacitance |
| `dt` | 0.025 | ms | Integration timestep |
| `v_threshold` | -20.0 | mV | Spike detection threshold |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/interneurons/martinotti_neuron.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` macro |
| NetworkRunner wired | `NeuronVariant::Martinotti` |
| `create_neuron("Martinotti")` | Yes |
| coverage tests | 7 (fire, no-fire, negative, adaptation, rebound, bounded, performance) |
| Pipeline integration test | `interneuron_population_create_step_reset`, `interneuron_mixed_network` |
| NaN/extreme input test | `all_models_nan_input_stays_finite`, `all_models_extreme_input_stays_finite` |
| Benchmark | `martinotti_1k_steps`: **505 us** (505 ns/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| martinotti_1k_steps | 505 us |
| Per step | 505 ns |

Slightly faster than SST (586 us) despite sharing the same gating scheme. The Martinotti model lacks the h-current (no Ih computation), saving one exp() evaluation and one current term per sub-step.

---

## Comparison with Related Models

| Property | Martinotti (this) | SST | PV+ | Pospischil LTS |
|----------|------------------|-----|-----|----------------|
| M-current | g_m=0.25 (very strong) | g_m=0.12 | No | Yes |
| T-type Ca2+ | Yes | Yes | No | Yes |
| h-current (Ih) | No | Yes | No | Optional |
| g_na | 40.0 | 50.0 | 35.0 | 50.0 |
| Capacitance | 0.8 | 1.0 | 1.0 | 1.0 |
| Adaptation strength | Very strong | Moderate | None | Moderate |
| Sub-steps | 4 | 4 | 50 | 4 |
| Per 1k steps | 505 us | 586 us | 4.35 ms | ~500 us |

The Martinotti model is a stripped-down SST variant: same Na+/K+ gating and T-type Ca2+, but no Ih and more than double the M-current conductance. The result is a cell dominated by adaptation rather than sag/rebound.

---

## Findings

1. **Very strong adaptation from g_m=0.25.** At more than twice the SST value, the M-current produces rapid firing-rate decay. Sustained input that drives SST at 40 Hz steady state will drive Martinotti to ~15-20 Hz after adaptation.
2. **No Ih means no sag.** Unlike SST, Martinotti lacks voltage sag during hyperpolarisation. Rebound spiking depends solely on T-type Ca2+ de-inactivation, making it weaker and more threshold-sensitive than SST rebound.
3. **Lower capacitance (0.8) compensates for lower g_na.** With g_na=40 (vs SST's 50) but C_m=0.8 (vs 1.0), the effective dV/dt during AP upstroke is comparable. The reduced capacitance also lowers rheobase.
4. **T-type Ca2+ identical to SST.** Same gating parameters (V_half=-57, k=6.2) and conductance (g_t=0.01). Rebound is possible but weaker without Ih to prime the T-current via hyperpolarisation-induced de-inactivation.
5. **NaN-safe after reset.** NaN input corrupts state, but reset() restores finite values.
