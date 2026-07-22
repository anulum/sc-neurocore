# PVFastSpikingNeuron

**Module:** `engine/src/neurons/interneurons/pv_fast_spiking.rs`
**Reference:** Wang & Buzsaki, J. Neurosci. 16(20), 1996 + Kv3.1 extension
**Family:** Hodgkin-Huxley variant, fast-spiking GABAergic interneuron
**State variables:** `v` (membrane potential), `h` (Na+ inactivation), `n` (K+ activation), `p` (Kv3.1 activation)

---

## Biological Context

Parvalbumin-positive (PV+) interneurons are the most abundant cortical inhibitory cell type (~40% of GABAergic neurons). They provide fast, precise perisomatic inhibition onto pyramidal cells, generating gamma oscillations (30-80 Hz) critical for sensory processing, attention, and working memory.

Key electrophysiological features:
- Narrow action potentials (~0.3 ms at half-width)
- High sustained firing rates (>200 Hz)
- No spike-frequency adaptation
- Low input resistance (~100 MOhm)
- Fast membrane time constant (~5 ms effective)

The narrow APs and high-frequency capability depend on Kv3.1 channels (KCNC1), which activate at depolarised potentials and deactivate rapidly, enabling fast repolarisation without afterhyperpolarisation-mediated delays.

---

## Equations

### Wang-Buzsaki core (Na+, K+, leak)

$$C_m \frac{dV}{dt} = -g_{Na} m_\infty^3 h (V - E_{Na}) - g_K n^4 (V - E_K) - g_L (V - E_L) - g_{Kv3} p (V - E_K) + I_{ext}$$

$$m_\infty = \frac{\alpha_m}{\alpha_m + \beta_m}$$

$$\frac{dh}{dt} = \phi (\alpha_h (1 - h) - \beta_h h)$$

$$\frac{dn}{dt} = \phi (\alpha_n (1 - n) - \beta_n n)$$

### Kv3.1 (fast-activating K+ for narrow APs)

$$p_\infty = \frac{1}{1 + \exp(-(V + 10)/10)}$$

$$\frac{dp}{dt} = \phi \frac{p_\infty - p}{1.0}$$

### Alpha/beta gating (standard HH)

$$\alpha_m = \frac{0.1(V + 35)}{1 - \exp(-(V + 35)/10)}, \quad \beta_m = 4 \exp(-(V + 60)/18)$$

$$\alpha_h = 0.07 \exp(-(V + 58)/20), \quad \beta_h = \frac{1}{1 + \exp(-(V + 28)/10)}$$

$$\alpha_n = \frac{0.01(V + 34)}{1 - \exp(-(V + 34)/10)}, \quad \beta_n = 0.125 \exp(-(V + 44)/80)$$

The kinetic scaling factor $\phi = 5$ accelerates all gating variables, producing the fast kinetics characteristic of PV+ cells.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `h` | 0.8 | — | Na+ inactivation gate |
| `n` | 0.1 | — | Delayed-rectifier K+ activation |
| `p` | 0.0 | — | Kv3.1 activation |
| `g_na` | 35.0 | mS/cm^2 | Na+ conductance |
| `g_k` | 9.0 | mS/cm^2 | Delayed-rectifier K+ |
| `g_kv3` | 5.0 | mS/cm^2 | Kv3.1 conductance |
| `g_l` | 0.1 | mS/cm^2 | Leak conductance |
| `e_na` | 55.0 | mV | Na+ reversal |
| `e_k` | -90.0 | mV | K+ reversal |
| `e_l` | -65.0 | mV | Leak reversal |
| `c_m` | 1.0 | uF/cm^2 | Membrane capacitance |
| `phi` | 5.0 | — | Kinetic scaling factor |
| `dt` | 0.01 | ms | Integration timestep |
| `v_threshold` | -20.0 | mV | Spike detection threshold |

Sub-stepping: 50 steps per call (0.5 ms real time per call).

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/interneurons/pv_fast_spiking.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` macro |
| NetworkRunner wired | `NeuronVariant::PVFastSpiking` |
| `create_neuron("PVFastSpiking")` | Yes |
| coverage tests | 7 (fire, no-fire, negative, high-rate, reset, bounded, performance) |
| Pipeline integration test | `interneuron_population_create_step_reset`, `interneuron_mixed_network` |
| NaN/extreme input test | `all_models_nan_input_stays_finite`, `all_models_extreme_input_stays_finite` |
| Benchmark | `pv_fs_1k_steps`: **4.35 ms** (4.35 us/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| pv_fs_1k_steps | 4.35 ms |
| Per step | 4.35 us |

The higher per-step cost vs simpler models (e.g. Lapicque at 2.1 ns) is due to 50 sub-steps with exp() evaluations per call. This is the cost of biophysical fidelity in the Wang-Buzsaki gating scheme.

---

## Comparison with Related Models

| Property | PV+ (this) | GolombFS | WangBuzsaki | Pospischil FS |
|----------|-----------|----------|-------------|---------------|
| Kv3.1 channel | Yes | Yes (Kv3) | No | No (M-current) |
| Gating scheme | Alpha/beta (WB) | Sigmoid/tau | Alpha/beta | Alpha/beta |
| Repetitive firing | Yes | Limited* | Yes | Yes |
| Sub-steps | 50 | 10 | 50 | 4 |
| Per step | 4.35 us | ~3 us | ~4 us | ~0.5 us |

*GolombFS with constant tau values enters depolarisation block at moderate current — addressed by using WB gating in this model.

---

## Findings

1. **WB gating essential for repetitive firing.** The original Golomb model (sigmoid activation + constant tau) enters depolarisation block after 1-2 spikes. Alpha/beta gating with phi=5 produces sustained high-frequency firing.
2. **Kv3.1 narrows APs.** Adding g_kv3=5.0 reduces spike width without affecting firing rate at moderate current.
3. **Firing threshold ~2.0 uA/cm^2.** Below this: no spikes. Above: sustained firing at increasing rate.
4. **NaN-safe after reset.** NaN input corrupts state, but reset() restores finite values.
