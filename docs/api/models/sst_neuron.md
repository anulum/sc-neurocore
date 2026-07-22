# SSTNeuron

**Module:** `engine/src/neurons/interneurons/sst_neuron.rs`
**Reference:** Pospischil et al., Biol. Cybern. 99(4-5), 2008 (LTS parameterisation)
**Family:** Hodgkin-Huxley variant, low-threshold spiking GABAergic interneuron
**State variables:** `v` (membrane potential), `m` (Na+ activation), `h` (Na+ inactivation), `n` (K+ activation), `p` (M-current activation), `s` (T-type Ca2+ inactivation), `r` (h-current activation)

---

## Biological Context

Somatostatin-positive (SST+) interneurons constitute ~30% of cortical GABAergic neurons. They are low-threshold spiking (LTS) cells that target distal dendrites of pyramidal neurons, providing frequency-dependent inhibition that shapes dendritic integration and cortical gain control.

Key electrophysiological features:
- Spike-frequency adaptation driven by M-current (Kv7)
- Rebound bursting via T-type Ca2+ channels (low-threshold)
- Voltage sag from h-current (Ih / HCN channels)
- Facilitating synapses onto pyramidal dendrites
- Moderate firing rates (~20-60 Hz sustained)
- Higher input resistance than PV+ cells

The interplay of M-current (adaptation), T-type Ca2+ (rebound), and Ih (sag) produces the characteristic LTS phenotype: initial burst followed by adapted regular firing, and post-inhibitory rebound spikes from hyperpolarised states.

---

## Equations

### Core currents (Na+, K+, leak)

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_M - I_T - I_h - I_L + I_{ext}$$

$$I_{Na} = g_{Na} m^3 h (V - E_{Na})$$

$$I_K = g_K n^4 (V - E_K)$$

### Na+ gating (Pospischil alpha/beta, $V_T = -56.2$ mV)

$$\alpha_m = \frac{-0.32 (V - V_T - 13)}{e^{-(V - V_T - 13)/4} - 1}, \quad \beta_m = \frac{0.28 (V - V_T - 17)}{e^{(V - V_T - 17)/5} - 1}$$

$$\alpha_h = 0.128 \, e^{-(V - V_T - 17)/18}, \quad \beta_h = \frac{4}{1 + e^{-(V - V_T - 40)/5}}$$

### K+ gating

$$\alpha_n = \frac{-0.032 (V - V_T - 15)}{e^{-(V - V_T - 15)/5} - 1}, \quad \beta_n = 0.5 \, e^{-(V - V_T - 10)/40}$$

### M-current (Kv7, slow K+ for adaptation)

$$p_\infty = \frac{1}{1 + e^{-(V + 35)/10}}$$

$$\tau_p = \frac{400}{3.3 \, e^{(V + 35)/20} + e^{-(V + 35)/20}}$$

$$I_M = g_M \, p \, (V - E_K)$$

### T-type Ca2+ (low-threshold burst)

$$m_{T,\infty} = \frac{1}{1 + e^{-(V + 57)/6.2}}$$

$$s_\infty = \frac{1}{1 + e^{(V + 81)/4}}, \quad \tau_s = 30 + \frac{200}{1 + e^{(V + 70)/5}}$$

$$I_T = g_T \, m_{T,\infty}^2 \, s \, (V - E_{Ca})$$

### h-current (Ih, sag)

$$r_\infty = \frac{1}{1 + e^{(V + 80)/10}}, \quad \tau_r = 100 + \frac{500}{e^{-(V + 70)/20} + e^{(V + 70)/20}}$$

$$I_h = g_h \, r \, (V - E_h)$$

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
| `r` | 0.1 | -- | h-current (Ih) activation |
| `g_na` | 50.0 | mS/cm^2 | Na+ conductance |
| `g_k` | 5.0 | mS/cm^2 | Delayed-rectifier K+ |
| `g_m` | 0.12 | mS/cm^2 | M-current (adaptation) |
| `g_t` | 0.01 | mS/cm^2 | T-type Ca2+ (minimal window current) |
| `g_h` | 0.02 | mS/cm^2 | h-current (sag) |
| `g_l` | 0.05 | mS/cm^2 | Leak conductance |
| `e_na` | 50.0 | mV | Na+ reversal |
| `e_k` | -90.0 | mV | K+ reversal |
| `e_ca` | 120.0 | mV | Ca2+ reversal |
| `e_h` | -40.0 | mV | h-current reversal |
| `e_l` | -65.0 | mV | Leak reversal |
| `c_m` | 1.0 | uF/cm^2 | Membrane capacitance |
| `dt` | 0.025 | ms | Integration timestep |
| `v_threshold` | -20.0 | mV | Spike detection threshold |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/interneurons/sst_neuron.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` macro |
| NetworkRunner wired | `NeuronVariant::SST` |
| `create_neuron("SST")` | Yes |
| coverage tests | 7 (fire, no-fire, negative, adaptation, rebound, bounded, performance) |
| Pipeline integration test | `interneuron_population_create_step_reset`, `interneuron_mixed_network` |
| NaN/extreme input test | `all_models_nan_input_stays_finite`, `all_models_extreme_input_stays_finite` |
| Benchmark | `sst_1k_steps`: **586 us** (586 ns/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| sst_1k_steps | 586 us |
| Per step | 586 ns |

The lower cost relative to PV+ (4.35 ms) comes from 4 sub-steps vs 50. Each sub-step evaluates 6 current terms with multiple exp() calls (M-current tau, T-type gating, Ih tau), but the reduced sub-step count dominates.

---

## Comparison with Related Models

| Property | SST (this) | Martinotti | PV+ | Pospischil LTS |
|----------|-----------|------------|-----|----------------|
| M-current | g_m=0.12 | g_m=0.25 | No | Yes |
| T-type Ca2+ | Yes | Yes | No | Yes |
| h-current (Ih) | Yes | No | No | Optional |
| Adaptation | Moderate | Strong | None | Moderate |
| Sub-steps | 4 | 4 | 50 | 4 |
| Per 1k steps | 586 us | 505 us | 4.35 ms | ~500 us |

SST and Martinotti share the same gating scheme (Pospischil Na+/K+) and both include M-current + T-type Ca2+. The SST model adds Ih (sag current) and uses weaker M-current (g_m=0.12 vs 0.25), producing less adaptation but enabling voltage sag and rebound behaviour absent in the Martinotti model.

---

## Findings

1. **Ih enables sag and rebound.** The h-current (g_h=0.02) produces measurable voltage sag during hyperpolarisation. Combined with T-type Ca2+, this supports post-inhibitory rebound spiking.
2. **M-current drives adaptation.** At g_m=0.12, the slow K+ current produces clear spike-frequency adaptation across the firing range without silencing the cell at moderate input.
3. **T-type window current is minimal.** With g_t=0.01, the T-type Ca2+ contributes primarily to rebound bursts; steady-state window current near rest is negligible and does not destabilise the resting potential.
4. **NaN-safe after reset.** NaN input corrupts state, but reset() restores finite values.
