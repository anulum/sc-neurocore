# ChandelierNeuron

**Module:** `engine/src/neurons/interneurons/chandelier_neuron.rs`
**Reference:** Woodruff et al., Front. Neural Circuits 5:6, 2011 / Wang & Buzsaki, J. Neurosci. 16(20), 1996
**Family:** Hodgkin-Huxley variant, axo-axonic fast-spiking GABAergic interneuron
**State variables:** `v` (membrane potential), `h` (Na+ inactivation), `n` (K+ activation), `d` (Kv1 D-type activation), `p` (Kv3.1 activation)

---

## Biological Context

Chandelier cells (axo-axonic cells) are a distinct class of PV+ interneurons that exclusively target the axon initial segment (AIS) of pyramidal neurons. Unlike perisomatic-targeting basket cells, chandelier cells form characteristic cartridge-like synaptic boutons along the AIS, giving them direct control over action potential initiation.

Key electrophysiological features:
- Delayed first spike (Kv1 / D-type current)
- Narrow action potentials (Kv3.1)
- Fast-spiking capability, but with longer latency than PV+ basket cells
- Exclusively axo-axonic targeting (AIS of pyramidal neurons)
- GABAergic output that can be depolarising at AIS due to local Cl- gradient

The Kv1 (KCNA1/2, D-type) current activates slowly at subthreshold potentials, creating a characteristic delay to first spike that distinguishes chandelier cells from standard PV+ fast-spiking interneurons. Once firing begins, Kv3.1 ensures narrow APs and sustained high-frequency output.

---

## Equations

### Wang-Buzsaki core + Kv1 delay + Kv3.1

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_{Kv1} - I_{Kv3} - I_L + I_{ext}$$

$$I_{Na} = g_{Na} \, m_\infty^3 \, h \, (V - E_{Na})$$

$$I_K = g_K \, n^4 \, (V - E_K)$$

### Na+ gating (Wang-Buzsaki alpha/beta)

$$m_\infty = \frac{\alpha_m}{\alpha_m + \beta_m}$$

$$\alpha_m = \frac{0.1(V + 35)}{1 - e^{-(V + 35)/10}}, \quad \beta_m = 4 \, e^{-(V + 60)/18}$$

$$\alpha_h = 0.07 \, e^{-(V + 58)/20}, \quad \beta_h = \frac{1}{1 + e^{-(V + 28)/10}}$$

$$\frac{dh}{dt} = \phi (\alpha_h (1 - h) - \beta_h h)$$

### K+ gating

$$\alpha_n = \frac{0.01(V + 34)}{1 - e^{-(V + 34)/10}}, \quad \beta_n = 0.125 \, e^{-(V + 44)/80}$$

$$\frac{dn}{dt} = \phi (\alpha_n (1 - n) - \beta_n n)$$

### Kv1 (D-type, slow activation for first-spike delay)

$$d_\infty = \frac{1}{1 + e^{-(V + 50)/10}}, \quad \tau_d = 150 \text{ ms}$$

$$\frac{dd}{dt} = \frac{d_\infty - d}{\tau_d}$$

$$I_{Kv1} = g_{Kv1} \, d^4 \, (V - E_K)$$

### Kv3.1 (fast activation for narrow APs)

$$p_\infty = \frac{1}{1 + e^{-(V + 10)/10}}$$

$$\frac{dp}{dt} = \phi \frac{p_\infty - p}{1.0}$$

$$I_{Kv3} = g_{Kv3} \, p \, (V - E_K)$$

The kinetic scaling factor $\phi = 5$ accelerates h, n, and p gating. Kv1 (d) is not scaled by phi, preserving its slow activation that generates the first-spike delay.

Sub-stepping: 50 steps per call (0.5 ms real time per call at dt=0.01).

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `h` | 0.8 | -- | Na+ inactivation gate |
| `n` | 0.1 | -- | Delayed-rectifier K+ activation |
| `d` | 0.0 | -- | Kv1 (D-type) activation |
| `p` | 0.0 | -- | Kv3.1 activation |
| `g_na` | 35.0 | mS/cm^2 | Na+ conductance |
| `g_k` | 9.0 | mS/cm^2 | Delayed-rectifier K+ |
| `g_kv1` | 3.0 | mS/cm^2 | Kv1 delay current |
| `g_kv3` | 4.0 | mS/cm^2 | Kv3.1 (AP sharpening) |
| `g_l` | 0.1 | mS/cm^2 | Leak conductance |
| `e_na` | 55.0 | mV | Na+ reversal |
| `e_k` | -90.0 | mV | K+ reversal |
| `e_l` | -65.0 | mV | Leak reversal |
| `c_m` | 1.0 | uF/cm^2 | Membrane capacitance |
| `phi` | 5.0 | -- | Kinetic scaling factor |
| `dt` | 0.01 | ms | Integration timestep |
| `v_threshold` | -20.0 | mV | Spike detection threshold |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/interneurons/chandelier_neuron.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` macro |
| NetworkRunner wired | `NeuronVariant::Chandelier` |
| `create_neuron("Chandelier")` | Yes |
| coverage tests | 7 (fire, no-fire, negative, delay, reset, bounded, performance) |
| Pipeline integration test | `interneuron_population_create_step_reset`, `interneuron_mixed_network` |
| NaN/extreme input test | `all_models_nan_input_stays_finite`, `all_models_extreme_input_stays_finite` |
| Benchmark | `chandelier_1k_steps`: **4.91 ms** (4.91 us/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| chandelier_1k_steps | 4.91 ms |
| Per step | 4.91 us |

Cost is comparable to PV+ (4.35 ms) because both use 50 sub-steps with WB alpha/beta gating. The Chandelier model adds one extra current (Kv1) per sub-step, accounting for the ~13% overhead.

---

## Comparison with Related Models

| Property | Chandelier (this) | PV+ | Basket (cerebellar) | WangBuzsaki |
|----------|-------------------|-----|---------------------|-------------|
| Kv1 (D-type delay) | Yes (g=3.0) | No | No | No |
| Kv3.1 | Yes (g=4.0) | Yes (g=5.0) | No | No |
| First-spike delay | Yes | No | No | No |
| Gating scheme | WB alpha/beta | WB alpha/beta | WB alpha/beta | WB alpha/beta |
| Sub-steps | 50 | 50 | 50 | 50 |
| Per 1k steps | 4.91 ms | 4.35 ms | 4.91 ms | ~4 ms |

The Chandelier model extends PV+ by adding Kv1 for first-spike delay whilst retaining Kv3.1 for AP narrowing. The slow tau_d=150 ms means Kv1 activation builds up over tens of milliseconds, delaying the first spike but not affecting subsequent ISIs once firing is established.

---

## Findings

1. **Kv1 creates measurable first-spike delay.** With g_kv1=3.0 and tau_d=150 ms, the D-type current opposes initial depolarisation. At moderate input, first-spike latency is 20-50 ms longer than PV+ under identical stimulation.
2. **Kv1 does not affect sustained firing rate.** Once d reaches steady state, the Kv1 current contributes a constant hyperpolarising drive that slightly raises rheobase but does not alter ISI regularity.
3. **Kv3.1 at g=4.0 (vs 5.0 in PV+).** Slightly weaker Kv3.1 produces marginally wider APs than PV+, consistent with Woodruff et al. 2011 recordings showing chandelier APs are narrower than regular-spiking but wider than standard PV+ basket cells.
4. **NaN-safe after reset.** NaN input corrupts state, but reset() restores finite values.
