# VIPNeuron

**Module:** `engine/src/neurons/interneurons/vip_neuron.rs`
**Reference:** Porter et al., J. Neurosci. 18(20), 1998 / Bhatt et al., J. Physiol. 597(3), 2019
**Family:** Hodgkin-Huxley variant, irregular-spiking GABAergic interneuron
**State variables:** `v` (membrane potential), `h` (Na+ inactivation), `n` (K+ activation), `a` (A-type K+ activation), `b` (A-type K+ inactivation)

---

## Biological Context

Vasoactive intestinal peptide-positive (VIP+) interneurons are a specialised disinhibitory cell class (~15% of cortical GABAergic neurons). They preferentially inhibit SST+ and PV+ interneurons, releasing pyramidal cells from inhibition. This disinhibitory motif is central to attentional gating and top-down modulation.

Key electrophysiological features:
- Irregular/accommodating firing pattern
- High input resistance (small soma, bipolar morphology)
- Prominent A-type K+ current (Kv4) causing first-spike delay and accommodation
- Low membrane capacitance (~0.5 uF/cm^2)
- Low rheobase due to high input resistance
- No sustained high-frequency firing

The A-type K+ current activates rapidly near threshold and inactivates slowly, opposing the first few spikes. This produces the characteristic irregular, accommodating pattern distinct from the sustained firing of PV+ or the adapted regular firing of SST+ cells.

---

## Equations

### Core currents

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_A - I_L + I_{ext}$$

### Na+ (sigmoid steady-state)

$$m_\infty = \frac{1}{1 + e^{-(V + 30)/9.5}}$$

$$h_\infty = \frac{1}{1 + e^{(V + 53)/7}}, \quad \tau_h = 0.37 + \frac{2.78}{1 + e^{(V + 40.5)/6}}$$

$$I_{Na} = g_{Na} \, m_\infty^3 \, h \, (V - E_{Na})$$

### Delayed-rectifier K+

$$n_\infty = \frac{1}{1 + e^{-(V + 30)/10}}, \quad \tau_n = 0.37 + \frac{1.85}{1 + e^{(V + 27)/15}}$$

$$I_K = g_K \, n^4 \, (V - E_K)$$

### A-type K+ (accommodation current, Kv4)

$$a_\infty = \frac{1}{1 + e^{-(V + 50)/20}}, \quad \tau_a = 5.0 \text{ ms}$$

$$b_\infty = \frac{1}{1 + e^{(V + 78)/6}}, \quad \tau_b = 50.0 \text{ ms}$$

$$I_A = g_A \, a^3 \, b \, (V - E_K)$$

Sub-stepping: 4 steps per call (0.1 ms real time per call at dt=0.025).

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `h` | 0.8 | -- | Na+ inactivation gate |
| `n` | 0.1 | -- | Delayed-rectifier K+ activation |
| `a` | 0.0 | -- | A-type K+ activation |
| `b` | 0.9 | -- | A-type K+ inactivation |
| `g_na` | 35.0 | mS/cm^2 | Na+ conductance |
| `g_k` | 6.0 | mS/cm^2 | Delayed-rectifier K+ |
| `g_a` | 8.0 | mS/cm^2 | A-type K+ (accommodation) |
| `g_l` | 0.01 | mS/cm^2 | Leak conductance (high Rin) |
| `e_na` | 55.0 | mV | Na+ reversal |
| `e_k` | -90.0 | mV | K+ reversal |
| `e_l` | -65.0 | mV | Leak reversal |
| `c_m` | 0.5 | uF/cm^2 | Membrane capacitance (small soma) |
| `dt` | 0.025 | ms | Integration timestep |
| `v_threshold` | -20.0 | mV | Spike detection threshold |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/interneurons/vip_neuron.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` macro |
| NetworkRunner wired | `NeuronVariant::VIP` |
| `create_neuron("VIP")` | Yes |
| coverage tests | 7 (fire, no-fire, negative, accommodation, reset, bounded, performance) |
| Pipeline integration test | `interneuron_population_create_step_reset`, `interneuron_mixed_network` |
| NaN/extreme input test | `all_models_nan_input_stays_finite`, `all_models_extreme_input_stays_finite` |
| Benchmark | `vip_1k_steps`: **351 us** (351 ns/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| vip_1k_steps | 351 us |
| Per step | 351 ns |

The lowest cost among the six interneuron models. Four sub-steps with only 4 current terms (no Ca2+, no Ih, no M-current) and sigmoid steady-state Na+ (no alpha/beta computation) make this the lightest HH-variant interneuron.

---

## Comparison with Related Models

| Property | VIP (this) | SST | PV+ | Connor-Stevens |
|----------|-----------|-----|-----|----------------|
| A-type K+ | g_a=8.0 | No | No | Yes |
| Adaptation mechanism | A-current | M-current | None | A-current |
| Capacitance | 0.5 | 1.0 | 1.0 | 1.0 |
| Leak conductance | 0.01 | 0.05 | 0.1 | 0.3 |
| Sub-steps | 4 | 4 | 50 | varies |
| Per 1k steps | 351 us | 586 us | 4.35 ms | ~400 us |

The VIP model shares A-type K+ with Connor-Stevens but uses a simpler sigmoid gating scheme and lower capacitance to capture the high-Rin, small-soma VIP phenotype. Unlike SST+ (M-current adaptation), VIP accommodation arises from transient A-current opposing threshold crossing.

---

## Findings

1. **A-current dominates firing pattern.** With g_a=8.0 (strongest conductance after Na+), the A-type K+ current delays first spike and produces accommodation. Reducing g_a below 4.0 converts firing to regular sustained.
2. **High input resistance from low g_l.** At g_l=0.01, the model has very high input resistance, consistent with in-vitro recordings of VIP+ cells. Small currents produce large voltage excursions.
3. **Low capacitance accelerates dynamics.** C_m=0.5 halves the membrane time constant relative to standard 1.0 models, producing faster voltage changes per unit current.
4. **NaN-safe after reset.** NaN input corrupts state, but reset() restores finite values.
