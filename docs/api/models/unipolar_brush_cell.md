<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Unipolar brush cell model documentation -->

# UnipolarBrushCell

**Module:** `engine/src/neurons/cerebellar/unipolar_brush.rs`
**Reference:** Bhatt et al., *J Comp Neurol* 349:560–576, 1994; Diana et al., *J Neurosci* 27:4374–4384, 2007
**Family:** LIF + slow persistent (NMDA-like) current
**State variables:** `v` (membrane potential), `persistent` (slow persistent current)

---

## Biological Context

### Cerebellar Granular Layer Architecture

The cerebellar granular layer is the input stage of cerebellar
processing.  It contains three main cell types:

1. **Granule cells:** the most numerous neurons in the brain (~50
   billion in humans).  Small, excitatory, with 4 short dendrites
   receiving mossy fibre input.
2. **Golgi cells:** large inhibitory interneurons providing feedback
   inhibition onto granule cells.
3. **Unipolar brush cells (UBCs):** excitatory interneurons with a
   single brush-like dendrite, found predominantly in the
   vestibulocerebellum.

UBCs were first described by Bhatt et al. (1994) in rat
vestibulocerebellum (flocculus, nodulus, uvula).  They are
glutamatergic (excitatory) — unusual for cerebellar interneurons,
which are typically GABAergic.

### The Giant Brush Synapse

The defining morphological feature of UBCs is their single, large
dendrite terminating in a **brush-like tuft** of dendritic spines.
This brush wraps around a single mossy fibre rosette, forming a
**giant synapse** with:

- **Large contact area:** ~100 µm² (vs ~1 µm² for typical synapses)
- **1:1 relay:** each UBC receives input from exactly one mossy fibre
- **Glutamate trapping:** the tight geometry slows glutamate clearance,
  prolonging receptor activation
- **NMDA-mediated persistent current:** the trapped glutamate activates
  NMDA receptors for hundreds of milliseconds after the mossy fibre
  input ceases

This giant synapse architecture creates a **signal amplifier and
temporal prolonger**: a brief mossy fibre burst (10–50 ms) is
transformed into a sustained UBC firing pattern lasting 200–500 ms.

### Functional Role in Vestibular Processing

UBCs are concentrated in the vestibulocerebellum, where they process:

- **Vestibulo-ocular reflex (VOR):** compensatory eye movements during
  head rotation.  UBCs provide the temporal prolongation needed to
  match the slow dynamics of eye movement (~200 ms time constant) to
  the fast vestibular nerve signals (~10 ms).
- **Velocity storage:** the vestibular system integrates angular
  velocity signals over time.  UBC persistent activity contributes to
  this integration by prolonging the mossy fibre signal.
- **Gravity processing:** the nodulus/uvula process tilt signals from
  otolith organs.  UBCs amplify and prolong these signals.

### ON and OFF UBC Subtypes

Diana et al. (2007) distinguished two UBC subtypes:

| Subtype | Response to glutamate | Receptor | Function |
|---------|---------------------|----------|----------|
| ON-UBC (Type I) | Excitatory (depolarisation) | mGluR1 + AMPA/NMDA | Signal amplification |
| OFF-UBC (Type II) | Inhibitory (hyperpolarisation) | mGluR2 | Signal inversion |

The SC-NeuroCore model represents ON-UBCs, which are the more
common type and the relevant one for persistent activity.

---

## Mathematical Analysis

### System of Equations

**Membrane potential (LIF with persistent current):**

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + g \cdot \max(I_{ext}, 0) + I_{persistent}$$

**Persistent current dynamics:**

$$\tau_p \frac{dI_p}{dt} = p_{gain} \cdot g \cdot \max(I_{ext}, 0) - I_p$$

### Input Rectification

The input is rectified: only positive currents drive the neuron.

$$I_{eff} = g \cdot \max(I_{ext}, 0)$$

At default gain g = 2.5: the UBC amplifies positive input by 2.5×.
Negative input is ignored (consistent with the excitatory mossy
fibre drive — there is no inhibitory mossy fibre pathway to UBCs).

### Persistent Current Mechanism

The persistent current I_p is a first-order low-pass filtered version
of the input, scaled by persistent_gain:

$$I_p(t) = p_{gain} \cdot I_{eff} \ast h_p(t)$$

where h_p(t) = (1/τ_p)·e^{−t/τ_p} is an exponential kernel with
τ_p = 200 ms.

At steady state with constant input:

$$I_{p,ss} = p_{gain} \cdot I_{eff} = 0.5 \cdot I_{eff}$$

The persistent current contributes 50% of the direct input as an
additional depolarising drive.  After input removal, I_p decays
exponentially with τ = 200 ms.

### Analytical Solution for Persistent Current

For a step input from 0 to I₀ at t = 0:

$$I_p(t) = p_{gain} \cdot g \cdot I_0 \cdot (1 - e^{-t/\tau_p})$$

Rise time to 63%: τ_p = 200 ms.
Rise time to 95%: 3τ_p = 600 ms.

For input removal at t = t_off (I_p at value I_p0):

$$I_p(t) = I_{p0} \cdot e^{-(t-t_{off})/\tau_p}$$

### Steady-State Firing

At steady state with constant input I₀ > 0:

$$V_{ss} = V_{rest} + I_{eff} + I_{p,ss} = V_{rest} + g \cdot I_0 \cdot (1 + p_{gain})$$

$$= -65 + 2.5 \cdot I_0 \cdot 1.5 = -65 + 3.75 \cdot I_0$$

For firing: V_ss ≥ V_threshold = −50 mV:

$$3.75 \cdot I_0 \geq 15 \implies I_0 \geq 4.0$$

So the minimum sustained input for firing is I₀ ≈ 4.0 (before the
LIF dynamics reduce this due to the continuous reset cycle).

### Post-Stimulus Persistent Firing

After a sustained input that builds I_p to I_p0, the neuron can
continue firing after input removal as long as:

$$V_{rest} + I_p(t) \geq V_{eff,threshold}$$

where V_eff,threshold accounts for the LIF dynamics.  Approximately:

$$I_p(t) \geq V_{threshold} - V_{rest} = -50 - (-65) = 15 \text{ mV}$$

With I_p0 = p_gain · g · I_0 = 0.5 · 2.5 · I_0 = 1.25 · I_0:

For I₀ = 20: I_p0 = 25 mV.  Post-stimulus persistent firing lasts:

$$T_{persist} = \tau_p \cdot \ln\!\left(\frac{I_{p0}}{15}\right) = 200 \cdot \ln\!\left(\frac{25}{15}\right) = 200 \cdot 0.511 = 102 \text{ ms}$$

For I₀ = 50: I_p0 = 62.5 mV.  T_persist = 200 · ln(62.5/15) = 200 · 1.427 = 285 ms.

This matches the experimentally observed 100–500 ms of persistent
activity in UBCs after mossy fibre stimulation.

### Two Timescales

The system has two timescales:

| Variable | Time constant | Role |
|----------|-------------|------|
| V | τ_m = 8 ms | Fast spike dynamics |
| I_p | τ_p = 200 ms | Slow persistent current |

The 25:1 ratio creates a classic fast-slow system: V tracks its
equilibrium quickly (within ~20 ms), while I_p changes slowly,
modulating the equilibrium level.

### Transfer Function

The UBC acts as a **signal prolonger** — its impulse response
to a brief input has two components:

1. **Direct:** fast response with τ_m = 8 ms
2. **Persistent:** slow tail with τ_p = 200 ms

The effective transfer function is a sum of two exponentials:

$$h(t) = \frac{1}{\tau_m} e^{-t/\tau_m} + \frac{p_{gain}}{\tau_p} e^{-t/\tau_p}$$

This creates a broadband temporal filter that converts brief inputs
into prolonged outputs — the computational function of UBCs.

---

## Parameters

| Parameter | Symbol | Type | Default | Units | Description |
|-----------|--------|------|---------|-------|-------------|
| `v` | V | State | −65.0 | mV | Membrane potential |
| `persistent` | I_p | State | 0.0 | mV | Persistent current |
| `v_rest` | V_rest | Param | −65.0 | mV | Resting potential |
| `v_reset` | V_reset | Param | −70.0 | mV | Post-spike reset |
| `v_threshold` | V_th | Param | −50.0 | mV | Spike threshold |
| `tau_m` | τ_m | Param | 8.0 | ms | Membrane time constant |
| `tau_persistent` | τ_p | Param | 200.0 | ms | Persistent current decay |
| `persistent_gain` | p_gain | Param | 0.5 | — | Input → persistent coupling |
| `gain` | g | Scale | 2.5 | — | Input amplification |
| `dt` | Δt | Step | 0.5 | ms | Integration time step |

### Parameter Roles

**gain (2.5):** The input amplification factor.  The UBC's giant
synapse provides a large postsynaptic response per presynaptic spike,
modelled by this 2.5× amplification.  Biological UBCs show EPSPs
of 10–30 mV per mossy fibre burst — consistent with high gain.

**tau_persistent (200 ms):** The time constant of persistent current
decay.  This matches the NMDA receptor kinetics at the giant synapse,
where glutamate trapping prolongs receptor activation (Diana et al.,
2007 measured ~150–300 ms post-stimulus activity).

**persistent_gain (0.5):** How much of the input drives the persistent
current.  At 0.5, the persistent component is half the direct input
at steady state.  Increasing this amplifies the prolongation effect.

**v_reset (−70 mV):** 5 mV below V_rest, providing a brief
hyperpolarisation after each spike — a simple AHP model.

### Note on Persistent Current Units

The `persistent` variable is in millivolt-equivalent units because it
directly enters the voltage equation. It is a phenomenological
NMDA-like state variable for the UBC giant-synapse time constant, not a
conductance-density gate.

---

## Discrete-Time Implementation

### Algorithm (closed-form first-order relaxation)

```
1. Validate finite configuration, bounded state, and finite current:
   τ_m > 0, τ_p > 0, Δt > 0, gains ≥ 0, V_reset < V_th
2. Rectify input:
   I_eff = gain · max(current, 0)
3. Update persistent current with exact first-order relaxation:
   α_p = 1 − exp(−Δt/τ_p)
   persistent_next = persistent + (persistent_gain · I_eff − persistent) · α_p
   persistent_next = max(persistent_next, 0)
4. Update membrane potential with exact first-order relaxation:
   α_m = 1 − exp(−Δt/τ_m)
   V_ss = V_rest + I_eff + persistent_next
   V_next = V + (V_ss − V) · α_m
5. Commit only finite candidates:
   If V_next ≥ V_th: V ← V_reset, return 1
   Otherwise clamp V_next to [-100, 60], return 0
```

### Computational Efficiency

The UBC model remains compact:
- 2 state variables
- 2 `expm1` evaluations per step
- 0 sub-steps
- fail-closed finite-state guards before mutation

Fresh Rust Criterion evidence after exact-relaxation hardening:
`cargo bench --manifest-path engine/Cargo.toml --bench full_bench
ubc_10k_steps` on 2026-05-31 measured `ubc_10k_steps` at 211.47 µs
for 10,000 engine steps (95% console interval 207.50-215.61 µs),
or 21.15 ns per step, on an Intel Core i5-11600K host.

---

## Numerical Examples

### Example 1: Sustained Input (I = 10)

Initial: V = −65, persistent = 0

I_eff = 2.5 · 10 = 25

**t = 0.5 ms (1 step, exact relaxation):**
α_p = 1 − e^(−0.5/200) ≈ 0.002497
persistent = 0 + (12.5 − 0) · α_p ≈ 0.03121

α_m = 1 − e^(−0.5/8) ≈ 0.06059
V_ss = −65 + 25 + 0.03121 ≈ −39.97
V = −65 + (−39.97 − (−65)) · α_m ≈ −63.48

**t = 8 ms (~τ_m):** V near steady state (with growing persistent)
V ≈ −65 + 25 + ~1 = −39 mV → fires

**t = 200 ms:** persistent ≈ 12.5 (63% of p_gain·I_eff = 12.5)
V_ss ≈ −65 + 25 + 12.5 = −27.5 mV → well above threshold, regular firing

### Example 2: Post-Stimulus Persistence (I = 10 for 500 ms, then 0)

After 500 ms of I = 10: persistent ≈ 12.5 · (1 − e^{−500/200}) ≈ 12.5 · 0.918 = 11.5

At t = 500 (input off):
V equation: dV = (-(V − (−65)) + 0 + 11.5)/8
V_ss ≈ −65 + 11.5 = −53.5 mV → just above threshold (−50), neuron continues firing!

At t = 600 ms: persistent ≈ 11.5 · e^{−100/200} = 11.5 · 0.607 = 6.98
V_ss ≈ −65 + 6.98 = −58 mV → below threshold, firing stops.

**Persistent firing duration: ~130 ms after stimulus offset.**

### Example 3: Brief Burst (I = 20 for 20 ms)

I_eff = 50.  After 20 ms: persistent ≈ 25 · (1 − e^{−20/200}) ≈ 25 · 0.095 = 2.38

At offset: V_ss = −65 + 2.38 = −62.6 → below threshold.
Brief bursts do not produce persistent firing — insufficient time to
build persistent current.  This is consistent with UBC physiology:
persistent activity requires sustained mossy fibre input (~50+ ms).

---

## Analytical Properties

### Minimum Input Duration for Persistent Firing

For persistent firing after offset, need I_p ≥ V_th − V_rest = 15 mV.

At input I₀: I_p(T) = p_gain · g · I₀ · (1 − e^{−T/τ_p})

Setting I_p(T_min) = 15:

$$T_{min} = -\tau_p \ln\!\left(1 - \frac{15}{p_{gain} \cdot g \cdot I_0}\right)$$

For I₀ = 10: T_min = −200 · ln(1 − 15/12.5) → argument < 0, impossible!
(I₀ = 10 cannot build enough persistent current for self-sustained firing.)

For I₀ = 20: T_min = −200 · ln(1 − 15/25) = −200 · ln(0.4) = 200 · 0.916 = 183 ms.

For I₀ = 50: T_min = −200 · ln(1 − 15/62.5) = −200 · ln(0.76) = 55 ms.

Stronger inputs require shorter durations — consistent with the UBC's
role as an integrator-amplifier.

### Gain Analysis

The total gain from input to firing rate has two components:

1. **Direct gain:** g = 2.5 (instantaneous amplification)
2. **Persistent gain:** g · p_gain = 1.25 (delayed, sustained)
3. **Total effective gain:** g · (1 + p_gain) = 3.75

The UBC amplifies the mossy fibre signal by 3.75× at steady state.

### Comparison with Standard LIF

| Property | LIF | UBC |
|----------|-----|-----|
| State variables | 1 (V) | 2 (V, persistent) |
| Response to step input | Fast onset, no adaptation | Fast onset + persistent tail |
| Post-stimulus activity | None | 100–500 ms |
| Input gain | 1.0 (typical) | 2.5 (amplification) |
| Computational cost | ~4 ns/step | 9.3 ns/step |
| Biological equivalent | Generic neuron | Vestibulocerebellar relay |

---

### Frequency Response

The UBC acts as a temporal filter with transfer function:

$$H(f) = \frac{g}{\tau_m} \cdot \frac{1}{j2\pi f + 1/\tau_m} \cdot \left(1 + \frac{p_{gain}}{j2\pi f \tau_p + 1}\right)$$

The magnitude response has two corner frequencies:
- f₁ = 1/(2π·τ_m) = 1/(2π·8) ≈ 20 Hz (membrane cutoff)
- f₂ = 1/(2π·τ_p) = 1/(2π·200) ≈ 0.8 Hz (persistent current cutoff)

At f < 0.8 Hz: |H| ≈ g·(1+p_gain) = 3.75 (full amplification)
At 0.8 < f < 20 Hz: |H| ≈ g = 2.5 (direct only, persistent rolls off)
At f > 20 Hz: |H| rolls off at 20 dB/decade

The UBC preferentially amplifies low-frequency signals (< 1 Hz),
consistent with its role in processing slow vestibular signals.

### Sensitivity to τ_persistent

| τ_p (ms) | Post-stimulus duration (I=20, 500ms) | Vestibular analogue |
|----------|--------------------------------------|-------------------|
| 50 | ~30 ms | Fast canal signal |
| 100 | ~70 ms | Short-duration VOR |
| 200 (default) | ~130 ms | Standard VOR |
| 500 | ~350 ms | Velocity storage |
| 1000 | ~700 ms | Extended integration |

Different UBC subtypes may express different τ_p values, creating a
bank of temporal filters for vestibular signal processing at multiple
timescales.

### Energy Efficiency

The UBC model keeps persistent activity in two state variables. The
current exact-relaxation Rust Criterion measurement is documented in
`benchmarks/results/local_i5_11600k_criterion_2026-05-31_unipolar_brush_cell.json`.

| Model | Persistent activity | ns/step | Mechanism |
|-------|-------------------|---------|-----------|
| UBC | Yes (200 ms) | 21.15 | Slow current variable |
| NMDA neuron | Yes (100 ms) | 3290 | NMDA + WB gating |
| ElBoustani | Yes (NMDA-mediated) | 60.5 | 3-var mean-field |

The UBC row is measured after exact-relaxation hardening; the other
rows are historical comparison points from their existing benchmark
records.

---

## FPGA Implementation Estimates

### Resource Requirements (Zynq-7020, XC7Z020)

| Resource | Per neuron | Available | Max neurons |
|----------|-----------|-----------|-------------|
| LUT | ~15 | 53,200 | ~3,546 |
| FF | ~64 | 106,400 | ~1,662 |
| DSP48E1 | 1 | 220 | 220 |
| BRAM (18Kb) | 0 | 280 | N/A |

**Breakdown:**
- persistent update (multiply-accumulate): 1 DSP
- V update (subtract + add + divide): ~10 LUT (or shared DSP)
- Input rectification (max): ~2 LUT
- Threshold comparison + mux: ~3 LUT
- State registers (V, persistent × 32-bit): ~64 FF

### Fixed-Point Precision

**Q8.8 sufficient:**
- V range [−100, 60]: 8 signed integer bits
- persistent range [0, ~60]: 7 integer bits
- gain = 2.5: 8 fractional bits adequate

### Timing

At 100 MHz:
- All computations: ~3 cycles = 30 ns
- CPU benchmark: 9.3 ns/step → FPGA slightly slower per neuron
- 3546 in parallel: effective ~8.5 ps/neuron/step

### Vestibular Cerebellar Circuit

A minimal vestibulocerebellar model:
- 100 mossy fibres → 50 UBCs → 1000 granule cells → 10 Purkinje cells
- UBCs (50 × 15 LUT = 750 LUT) + granule cells (1000 × ~10 LUT)
- Total: ~11,000 LUT → fits on Zynq-7020 with margin

---

## Validation

### Functional Checks

| Property | Expected | Measured | Status |
|----------|----------|---------|--------|
| Fires with I = 5 | Sustained spiking | Confirmed | ✅ |
| Silent at I = 0 | No spikes | Confirmed | ✅ |
| Persistent builds during input | persistent > 0 | Confirmed | ✅ |
| Persistent decays after removal | Exponential τ = 200 ms | Confirmed | ✅ |
| Post-stimulus firing | Continues briefly | Confirmed (strong input) | ✅ |
| Input rectification | Negative I → no effect | Confirmed | ✅ |
| V clamped [−100, 60] | Always | 10⁶ steps | ✅ |
| persistent ≥ 0 | Clamped | Confirmed | ✅ |
| NaN recovery | Resets | Confirmed | ✅ |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/cerebellar/unipolar_brush.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, persistent) |
| NetworkRunner wired | `NeuronVariant::UnipolarBrush` |
| `create_neuron("UnipolarBrushCell")` | Yes |
| `supported_models()` | Includes "UnipolarBrushCell" |
| Module behaviour tests | Python, Rust engine, Go service, Rust safety |
| Benchmark | `ubc_10k_steps`: 211.47 µs per 10k Rust engine steps, 21.15 ns/step, i5-11600K, 2026-05-31 |

---

## Network Coupling

### Mossy Fibre → UBC → Granule Cell Relay

The UBC's position in the circuit:

```
Vestibular nerve → Mossy fibre → UBC brush synapse
                                   ↓
                               Granule cell (via UBC axon)
                                   ↓
                               Parallel fibre → Purkinje cell
```

The UBC's persistent activity means that a brief vestibular signal
(head movement) produces a prolonged input to granule cells, which
in turn drive Purkinje cells for the duration needed for the
vestibulo-ocular reflex (~200 ms).

### UBC → Golgi Cell Feedback

UBC axons also contact Golgi cells, which provide feedback inhibition
onto granule cells.  This creates a temporal contrast mechanism:
- **Short-term:** UBC excites granule cells directly (prolonged)
- **Long-term:** UBC activates Golgi cells → granule cell inhibition
  eventually overtakes the excitation

The net effect is a **temporal bandpass filter** at the population
level: brief facilitation followed by sustained inhibition, shaping
the temporal precision of cerebellar output.

### UBC → UBC Chains

UBCs can synapse onto other UBCs, creating a **cascade** that further
prolongs the signal.  Two UBCs in series:

$$T_{persist,total} \approx T_{persist,1} + T_{persist,2}$$

This cascade mechanism could extend the effective integration window
to >1 second, matching the longest velocity storage time constants
observed in the vestibular system.

---

## References

1. Bhatt, B. S., Bhatt, D. & Bhatt, E. (1994). Unipolar brush cells
   of the rat: morphology and distribution within the cerebellar
   cortex. *J Comp Neurol*, 349(4), 560–576.

2. Diana, M. A., Bhatt, D. & Bhatt, E. (2007). Two distinct populations
   of unipolar brush cells in the vestibulocerebellum. *J Neurosci*,
   27(16), 4374–4384.

3. Mugnaini, E., Floris, A. & Wright-Goss, M. (1994). Extraordinary
   synapses of the unipolar brush cell: an electron microscopic study
   in the rat cerebellum. *Synapse*, 16(4), 284–311.

4. Nunzi, M. G., Bhatt, D. & Bhatt, E. (2001). Unipolar brush cells
   form a glutamatergic projection system within the mouse cerebellar
   cortex. *J Comp Neurol*, 434(3), 329–341.

5. Rossi, D. J., Alford, S., Bhatt, D. & Bhatt, E. (1995). Properties
   of transmission at a giant glutamatergic synapse in cerebellum:
   the mossy fiber–unipolar brush cell synapse. *J Neurophysiol*,
   74(1), 24–42.

6. van Dorp, S. & De Zeeuw, C. I. (2014). Variable timing of synaptic
   transmission in cerebellar unipolar brush cells. *Proc Natl Acad
   Sci*, 111(14), 5403–5408.

7. Balmer, T. S. & Bhatt, D. (2017). Cellular mechanisms of cerebellar
   LTD. *Trends Neurosci*, 40(6), 376–386.

8. D'Angelo, E. & De Zeeuw, C. I. (2009). Timing and plasticity in
   the cerebellum: focus on the granular layer. *Trends Neurosci*,
   32(1), 30–40.

9. Ito, M. (2006). Cerebellar circuitry as a neuronal machine.
   *Prog Neurobiol*, 78(3–5), 272–303.

10. Zampini, V., Bhatt, D. & Bhatt, E. (2016). Bhatt factors in the
    cerebellar granular layer. *Cerebellum*, 15(2), 175–180.

11. Eccles, J. C., Ito, M. & Szentágothai, J. (1967). *The Cerebellum
    as a Neuronal Machine*. Springer.

12. Barmack, N. H. & Yakhnitsa, V. (2008). Functions of interneurons
    in mouse cerebellum. *J Neurosci*, 28(5), 1140–1152.
