# CerebellarBasketNeuron

**Module:** `engine/src/neurons/interneurons/cerebellar_basket_neuron.rs`
**Reference:** Midtgaard, J. Physiol. 457, 1992 / Hausser & Clark, Neuron 19(3), 1997 / Wang & Buzsaki, J. Neurosci. 16(20), 1996
**Family:** Hodgkin-Huxley variant, perisomatic-targeting cerebellar interneuron
**State variables:** `v` (membrane potential), `h` (Na+ inactivation), `n` (K+ activation), `a` (A-type K+ activation), `b` (A-type K+ inactivation), `ca` (intracellular Ca2+ concentration)

---

## Biological Context

Cerebellar basket cells are GABAergic interneurons of the molecular layer that provide perisomatic inhibition onto Purkinje cell somata and proximal dendrites. They form the characteristic basket-like plexus (pinceau) around Purkinje cell bodies and AIS, exerting powerful feed-forward inhibition in the cerebellar cortex.

Key electrophysiological features:
- Fast spiking with pronounced afterhyperpolarisation (AHP)
- A-type K+ current (transient outward, Kv4-family)
- Ca2+-activated K+ current driving medium AHP
- Intracellular Ca2+ accumulation during depolarisation
- Distinct from cortical PV+ by presence of A-current and Ca2+-dependent AHP
- Firing rates up to ~150 Hz

The combination of A-type K+ (first-spike modulation) and Ca2+-activated K+ (AHP) produces a firing pattern that differs from cortical PV+ cells: slightly wider APs, more prominent AHP, and moderate spike-frequency adaptation at high drive due to Ca2+ accumulation.

---

## Equations

### Wang-Buzsaki core + A-type K+ + Ca2+-activated K+

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_A - I_{KCa} - I_L + I_{ext}$$

$$I_{Na} = g_{Na} \, m_\infty^3 \, h \, (V - E_{Na})$$

$$I_K = g_K \, n^4 \, (V - E_K)$$

### Na+ and K+ gating (Wang-Buzsaki alpha/beta)

$$m_\infty = \frac{\alpha_m}{\alpha_m + \beta_m}$$

$$\alpha_m = \frac{0.1(V + 35)}{1 - e^{-(V + 35)/10}}, \quad \beta_m = 4 \, e^{-(V + 60)/18}$$

$$\alpha_h = 0.07 \, e^{-(V + 58)/20}, \quad \beta_h = \frac{1}{1 + e^{-(V + 28)/10}}$$

$$\frac{dh}{dt} = \phi (\alpha_h (1 - h) - \beta_h h)$$

$$\alpha_n = \frac{0.01(V + 34)}{1 - e^{-(V + 34)/10}}, \quad \beta_n = 0.125 \, e^{-(V + 44)/80}$$

$$\frac{dn}{dt} = \phi (\alpha_n (1 - n) - \beta_n n)$$

### A-type K+ (cerebellar, transient outward)

$$a_\infty = \frac{1}{1 + e^{-(V + 45)/15}}, \quad \frac{da}{dt} = \phi \frac{a_\infty - a}{5.0}$$

$$b_\infty = \frac{1}{1 + e^{(V + 75)/8}}, \quad \frac{db}{dt} = \frac{b_\infty - b}{50.0}$$

$$I_A = g_A \, a^3 \, b \, (V - E_K)$$

### Ca2+-activated K+ (AHP)

$$q_\infty = \frac{[Ca^{2+}]}{[Ca^{2+}] + 0.2}$$

$$I_{KCa} = g_{KCa} \, q_\infty \, (V - E_K)$$

### Intracellular Ca2+ dynamics

$$\frac{d[Ca^{2+}]}{dt} = -\frac{[Ca^{2+}]}{80} + I_{Ca,entry}$$

$$I_{Ca,entry} = \begin{cases} 0.01 (V + 20) & \text{if } V > -20 \text{ mV} \\ 0 & \text{otherwise} \end{cases}$$

The Ca2+ entry is a simplified voltage-gated term: depolarisation above -20 mV drives Ca2+ influx proportional to $(V + 20)$. Decay time constant of 80 ms represents pump/buffering clearance.

Sub-stepping: 50 steps per call (0.5 ms real time per call at dt=0.01).

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `h` | 0.8 | -- | Na+ inactivation gate |
| `n` | 0.1 | -- | Delayed-rectifier K+ activation |
| `a` | 0.0 | -- | A-type K+ activation |
| `b` | 0.9 | -- | A-type K+ inactivation |
| `ca` | 0.05 | uM | Intracellular Ca2+ concentration |
| `g_na` | 35.0 | mS/cm^2 | Na+ conductance |
| `g_k` | 9.0 | mS/cm^2 | Delayed-rectifier K+ |
| `g_a` | 3.0 | mS/cm^2 | A-type K+ (transient outward) |
| `g_kca` | 2.0 | mS/cm^2 | Ca2+-activated K+ (AHP) |
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
| Rust implementation | `engine/src/neurons/interneurons/cerebellar_basket_neuron.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` macro |
| NetworkRunner wired | `NeuronVariant::CerebellarBasket` |
| `create_neuron("CerebellarBasket")` | Yes |
| coverage tests | 7 (fire, no-fire, negative, AHP, reset, bounded, performance) |
| Pipeline integration test | `interneuron_population_create_step_reset`, `interneuron_mixed_network` |
| NaN/extreme input test | `all_models_nan_input_stays_finite`, `all_models_extreme_input_stays_finite` |
| Benchmark | `cerebellar_basket_1k_steps`: **4.91 ms** (4.91 us/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| cerebellar_basket_1k_steps | 4.91 ms |
| Per step | 4.91 us |

Same cost class as PV+ and Chandelier (50 sub-steps with WB gating). The additional A-type and Ca2+-activated K+ currents plus Ca2+ dynamics add ~13% overhead relative to PV+ (4.35 ms), matching the Chandelier model.

---

## Comparison with Related Models

| Property | Basket (this) | PV+ | Chandelier | VIP |
|----------|--------------|-----|------------|-----|
| A-type K+ | Yes (g=3.0) | No | No | Yes (g=8.0) |
| Ca2+-activated K+ | Yes (g=2.0) | No | No | No |
| Ca2+ dynamics | Yes | No | No | No |
| Kv3.1 | No | Yes | Yes | No |
| AHP mechanism | Ca2+-dependent | Minimal | Minimal | None |
| Sub-steps | 50 | 50 | 50 | 4 |
| Per 1k steps | 4.91 ms | 4.35 ms | 4.91 ms | 351 us |

The cerebellar basket model is the only interneuron with explicit intracellular Ca2+ tracking and Ca2+-activated K+. The A-type K+ differs from VIP (different half-activation: -45 vs -50 mV; different slope: 15 vs 20) reflecting cerebellar vs cortical channel subtypes.

---

## Findings

1. **Ca2+-activated K+ drives AHP.** With g_kca=2.0, Ca2+ accumulation during spike trains activates the KCa current, producing a medium AHP that increases with firing rate. This creates mild frequency-dependent adaptation absent in PV+.
2. **Ca2+ entry threshold at -20 mV.** The simplified voltage-gated Ca2+ entry activates only during APs (threshold crossing coincides with v_threshold=-20). Subthreshold activity does not accumulate Ca2+.
3. **Ca2+ decay (tau=80 ms) sets AHP duration.** The 80 ms clearance constant means AHP influence persists for ~200-300 ms after a burst, consistent with Midtgaard 1992 recordings.
4. **A-type K+ modulates first spike.** At g_a=3.0 (weaker than VIP's 8.0), the A-current provides modest first-spike delay without dominating the firing pattern.
5. **NaN-safe after reset.** NaN input corrupts state, but reset() restores finite values. Ca2+ is clamped to non-negative.
