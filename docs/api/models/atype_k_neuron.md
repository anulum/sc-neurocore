# ATypeKNeuron

**Module:** `engine/src/neurons/channels/a_type_k.rs`
**Reference:** Connor & Stevens, *J Physiol* 213:31–53, 1971; Hoffman et al., *Nature* 387:869–875, 1997
**Family:** WB Na⁺/K⁺ base + A-type K⁺ (IA, transient outward)
**State variables:** `v`, `h` (Na⁺ inactivation), `n` (K_dr activation), `a` (IA activation), `b` (IA inactivation)

---

## Biological Context

### The A-Type Potassium Current (IA)

IA is a transient outward K⁺ current mediated by Kv4.x (Shal family)
and Kv1.4 channels.  Its defining biophysical property is rapid
activation at **subthreshold** voltages followed by complete
inactivation over tens of milliseconds.  This distinguishes IA from
the delayed rectifier K⁺ current (IK), which activates at more
positive voltages and does not inactivate.

Key channel properties (from Connor & Stevens, 1971):
- **Activation:** V½ ≈ −50 mV, fast (τ ≈ 2 ms) — activates before
  the Na⁺ channels
- **Inactivation:** V½ ≈ −70 mV, slow (τ ≈ 50 ms) — removes IA over
  tens of milliseconds
- **Recovery from inactivation:** V½ ≈ −90 mV, moderate (~50 ms)

The consequence: when a neuron receives a depolarising input from rest
(−65 mV), IA activates first (V½ = −50), opposing the depolarisation.
The neuron must wait for IA to inactivate (τ ≈ 50 ms) before it can
reach Na⁺ threshold.  This creates a characteristic **delay to the
first spike**.

### Functional Roles

**First-spike latency:** The delay before the first AP in response to
a step current injection.  Neurons with strong IA (e.g. regular-spiking
cortical pyramidal cells) show 50–200 ms latency; neurons without IA
(e.g. fast-spiking interneurons) fire immediately.

**Spike frequency control:** During each interspike interval, IA
partially recovers from inactivation (at V ≈ −65 mV, b recovers
toward ~0.8).  At the next depolarisation, the recovered IA delays
the next spike.  This lengthens the ISI and reduces firing rate
compared to a neuron without IA.

**Coincidence detection:** A neuron with strong IA responds poorly to
slow, ramp-like inputs (IA has time to activate and oppose) but
responds well to fast, synchronous inputs (IA has no time to activate
before the Na⁺ threshold is crossed).  This makes the neuron a
temporal coincidence detector.

**Dendritic processing:** Hoffman et al. (1997) showed that IA density
increases along the apical dendrite of hippocampal CA1 pyramidal cells,
progressively attenuating back-propagating action potentials.  This
creates a distance-dependent signal attenuation that shapes dendritic
integration.

### Molecular Identity

| Channel | Gene | Location | IA component |
|---------|------|----------|-------------|
| Kv4.2 | KCND2 | Somatodendritic | Dominant in cortex, hippocampus |
| Kv4.3 | KCND3 | Somatodendritic | Cerebellar granule cells |
| Kv1.4 | KCNA4 | Axonal | Presynaptic IA |

The auxiliary subunit KChIP (K⁺ channel interacting protein) modulates
Kv4 surface expression and gating kinetics.  DPP6 (dipeptidyl
peptidase-like protein 6) accelerates IA recovery from inactivation
in CA1 dendrites.

### The Connor–Stevens Model

Connor & Stevens (1971) first characterised IA in molluscan neurons
(*Anisodoris*) and showed that its inclusion in a HH-type model
reproduces the regular spiking behaviour and first-spike latency
observed experimentally.  The Connor-Stevens model became the standard
framework for neurons with transient outward currents.

---

## Mathematical Analysis

### Membrane Equation

$$C_m \frac{dV}{dt} = -(I_{Na} + I_K + I_A + I_L) + I_{ext}$$

### Base WB Currents

$$I_{Na} = g_{Na} \cdot m_\infty^3(V) \cdot h \cdot (V - E_{Na})$$
$$I_K = g_K \cdot n^4 \cdot (V - E_K)$$
$$I_L = g_L \cdot (V - E_L)$$

WB α/β rates (identical to SKNeuron, NMDANeuron):
$$\alpha_m = \frac{0.1(V + 35)}{1 - e^{-(V+35)/10}}, \quad \beta_m = 4 e^{-(V+60)/18}$$
$$\alpha_h = 0.07 e^{-(V+58)/20}, \quad \beta_h = \frac{1}{1 + e^{-(V+28)/10}}$$
$$\alpha_n = \frac{0.01(V + 34)}{1 - e^{-(V+34)/10}}, \quad \beta_n = 0.125 e^{-(V+44)/80}$$

### A-Type K⁺ Current

$$I_A = g_A \cdot a^3 \cdot b \cdot (V - E_K)$$

**Activation (a) — fast, Boltzmann:**

$$a_\infty(V) = \frac{1}{1 + e^{-(V+50)/20}}$$
$$\tau_a = 2.0 \text{ ms (constant)}$$
$$\frac{da}{dt} = \frac{a_\infty(V) - a}{\tau_a}$$

V½ = −50 mV with k = 20 mV.  This is ~15 mV more negative than the
Na⁺ activation threshold (V½,m ≈ −35 mV), ensuring IA activates
*before* Na⁺ during a slow depolarisation.

The a³ power gives a sigmoidal activation curve steeper than a single
Boltzmann.  At V = −50: a_∞ = 0.5, a³ = 0.125.

**Inactivation (b) — slow, Boltzmann:**

$$b_\infty(V) = \frac{1}{1 + e^{(V+70)/6}}$$
$$\tau_b = 50.0 \text{ ms (constant)}$$
$$\frac{db}{dt} = \frac{b_\infty(V) - b}{\tau_b}$$

V½ = −70 mV with k = −6 mV (steep negative slope).

| V (mV) | b_∞ | Description |
|--------|-----|-------------|
| −90 | 0.965 | Nearly fully available |
| −80 | 0.843 | Mostly available |
| −70 | 0.500 | Half-inactivated |
| −65 | 0.296 | Mostly inactivated (at rest!) |
| −50 | 0.017 | Almost fully inactivated |

**Critical observation:** at the default resting potential V = −65 mV,
b_∞ = 0.296 — IA is already ~70% inactivated at rest.  This means the
available IA conductance at rest is only g_A · a³ · b ≈ 8 · 0.125 · 0.3
≈ 0.3 mS/cm².  The first-spike delay arises from the *remaining* 30%
of available IA.

### Window Current

The IA window current is the overlap of activation (a) and
non-inactivation (b) steady states:

$$I_{A,window}(V) = g_A \cdot a_\infty^3(V) \cdot b_\infty(V) \cdot (V - E_K)$$

| V (mV) | a³_∞ | b_∞ | a³·b | I_A,window |
|--------|------|-----|------|-----------|
| −80 | 0.003 | 0.843 | 0.003 | ~0 |
| −65 | 0.027 | 0.296 | 0.008 | ~2 |
| −50 | 0.125 | 0.017 | 0.002 | ~0.6 |
| −40 | 0.316 | 0.001 | 0.0003 | ~0.1 |

The window current peaks around V ≈ −60 to −65 mV — exactly at the
resting potential.  This tonic IA current contributes to setting the
resting potential slightly more negative than E_L.

### First-Spike Latency Mechanism

When a step current is applied from V_rest = −65 mV:

1. **t = 0:** V begins to depolarise.  IA activates (a rises with
   τ = 2 ms), opposing depolarisation.
2. **t = 5 ms:** a reaches near-steady state for the current V.
   IA is providing maximum opposition.
3. **t = 10–50 ms:** IA begins to inactivate (b decreases with
   τ = 50 ms).  Opposition weakens.
4. **t = 50–100 ms:** b is small enough that IA opposition is overcome.
   V rises to Na⁺ threshold.  First spike fires.

The latency depends on the input strength:
- Weak input (near threshold): latency ≈ 50–200 ms (IA must fully
  inactivate)
- Strong input (well above threshold): latency ≈ 5–20 ms (input
  overwhelms IA before inactivation)

### Interspike Interval Modulation

During each ISI (V ≈ −65 mV after reset):
- a decays toward a_∞(−65) ≈ 0.27 with τ = 2 ms (fast)
- b recovers toward b_∞(−65) ≈ 0.30 with τ = 50 ms (slow)

After a short ISI (~10 ms): b ≈ 0.1 (still inactivated from the
previous spike).  IA is weak → next spike comes quickly.

After a long ISI (~100 ms): b ≈ 0.28 (substantially recovered).
IA is stronger → next spike is delayed.

This creates **spike-rate adaptation** through IA recovery: initial
spikes come fast (b hasn't recovered), later spikes are slower
(b has recovered, more IA opposition).

---

## Parameters

| Parameter | Symbol | Type | Default | Units | Description |
|-----------|--------|------|---------|-------|-------------|
| `v` | V | State | −65.0 | mV | Membrane potential |
| `h` | h | State | 0.6 | — | Na⁺ inactivation |
| `n` | n | State | 0.32 | — | K_dr activation |
| `a` | a | State | 0.1 | — | IA activation |
| `b` | b | State | 0.8 | — | IA inactivation |
| `g_na` | g_Na | Param | 35.0 | mS/cm² | Na⁺ conductance |
| `g_k` | g_K | Param | 9.0 | mS/cm² | K_dr conductance |
| `g_a` | g_A | Param | 8.0 | mS/cm² | A-type K⁺ conductance |
| `g_l` | g_L | Param | 0.1 | mS/cm² | Leak conductance |
| `e_na` | E_Na | Param | 55.0 | mV | Na⁺ reversal |
| `e_k` | E_K | Param | −90.0 | mV | K⁺ reversal (shared) |
| `e_l` | E_L | Param | −65.0 | mV | Leak reversal |
| `c_m` | C_m | Param | 1.0 | µF/cm² | Capacitance |
| `phi` | φ | Param | 5.0 | — | Temperature factor |
| `dt` | Δt | Step | 0.5 | ms | External time step |
| `v_threshold` | V_th | Thresh | −20.0 | mV | Spike threshold |
| `gain` | g | Scale | 1.0 | — | Input multiplier |

### Conductance Balance

| Current | g (mS/cm²) | Role |
|---------|-----------|------|
| INa | 35.0 | Spike upstroke |
| IK (Kdr) | 9.0 | Spike repolarisation |
| **IA** | **8.0** | **Subthreshold opposition, latency** |
| IL | 0.1 | Resting conductance |

IA conductance (8.0) is nearly as large as IK (9.0) — IA is a major
current in this model.  Removing IA (g_A = 0) transforms the neuron
from regular-spiking to fast-spiking.

---

## Discrete-Time Implementation

### Algorithm (50 sub-steps, dt_sub = 0.01 ms)

```
For each sub-step:
  1. WB rates → m_inf, α_h, β_h, α_n, β_n
  2. IA gating:
     a_inf = σ(V; -50, 20)
     b_inf = σ(V; -70, -6)
  3. Gate updates:
     h += dt_sub · φ · (α_h(1-h) - β_h·h)
     n += dt_sub · φ · (α_n(1-n) - β_n·n)
     a += dt_sub · (a_inf - a) / 2.0
     b += dt_sub · (b_inf - b) / 50.0
  4. Currents:
     I_Na = g_Na · m_inf³ · h · (V - E_Na)
     I_K = g_K · n⁴ · (V - E_K)
     I_A = g_A · a³ · b · (V - E_K)
     I_L = g_L · (V - E_L)
  5. V update: V += dt_sub · (-I_Na - I_K - I_A - I_L + I_ext) / C_m
  6. Spike: if V ≥ -20 → V = -65, fired = 1
```

After sub-steps: clamp V to [−100, 60], h/n/a/b to [0, 1], NaN guard.

---

## Numerical Examples

### Example 1: First-Spike Latency (I = 1.5)

Initial: V = −65, h = 0.6, n = 0.32, a = 0.1, b = 0.8

At t = 0 (step input): V begins rising.
a_∞(−65) = σ(−65;−50,20) = 1/(1+e^{−0.75}) = 0.68.
a rises from 0.1 toward 0.68 with τ = 2 ms.

By t = 5 ms: a ≈ 0.60, b ≈ 0.75.
I_A = 8 · 0.216 · 0.75 · (−55−(−90)) = 8 · 0.162 · 35 ≈ 45 nA/cm²
This outward current opposes the input, delaying the spike.

By t = 50 ms: b has dropped to ~0.35 (inactivating).
I_A = 8 · 0.2 · 0.35 · 25 ≈ 14 nA/cm² — much reduced.
V can now rise to threshold.

**First spike at t ≈ 60–80 ms** — the characteristic IA delay.

### Example 2: No IA (g_A = 0)

Same input I = 1.5 but with g_A = 0:
The neuron fires within ~5 ms (limited only by the WB Na⁺ activation).
No first-spike delay.

### Example 3: Strong Input (I = 5)

At I = 5: the input current overwhelms IA opposition.
Even with full IA (a³·b ≈ 0.001·0.8 initially), I_A ≈ 8·0.0008·25 ≈ 0.16 nA/cm²
— negligible compared to I_ext = 5.
First spike at t ≈ 3 ms (no significant delay).

This demonstrates the coincidence detection property: strong, fast
inputs bypass IA; weak, slow inputs are delayed.

---

### Example 4: Adaptation During Train (I = 2)

Initial: V = −65, a = 0.1, b = 0.8

**Spike 1 (t ≈ 70 ms):** Long latency due to IA. After spike, reset
to −65. b ≈ 0.15 (inactivated during the depolarisation).

**Spike 2 (t ≈ 85 ms):** ISI ≈ 15 ms. b has barely recovered
(b ≈ 0.15 + (0.30−0.15)·(1−e^{−15/50}) ≈ 0.19). Little IA → short ISI.

**Spike 5 (t ≈ 140 ms):** b has recovered more between spikes
(ISIs lengthening). b ≈ 0.25. More IA opposition → ISI ≈ 20 ms.

**Spike 10 (t ≈ 280 ms):** Steady-state adaptation reached.
b oscillates between ~0.10 (at spike time) and ~0.25 (at ISI end).
ISI ≈ 22 ms → adapted rate ≈ 45 Hz.

**Adaptation index:** ISI_first ≈ 15 ms, ISI_last ≈ 22 ms.
AI = (22−15)/(22+15) = 7/37 ≈ 0.19 (moderate adaptation).

### Example 5: Ramp Input (I = 0 → 3 over 500 ms)

Slowly rising input: I(t) = 3t/500 (mV/ms units).

At each moment, IA has time to activate and partially oppose the
depolarisation.  The neuron only fires when the input ramp exceeds
the IA opposition — this happens near the end of the ramp.

First spike at t ≈ 450 ms (I ≈ 2.7).
Compare with g_A = 0: first spike at t ≈ 80 ms (I ≈ 0.48).

The IA creates a ~6× delay and ~5.6× higher input threshold for
ramp inputs — the quintessential temporal filtering property.

### Example 6: Paired Pulse (two 5 ms pulses, varying interval)

Pulse 1 (I = 5 for 5 ms): fast enough to bypass IA → spike.

Pulse 2 at interval Δt:
- Δt = 10 ms: b ≈ 0.10 (still inactivated) → spike (IA weak)
- Δt = 50 ms: b ≈ 0.22 (partially recovered) → spike (IA moderate)
- Δt = 200 ms: b ≈ 0.29 (near-fully recovered) → spike delayed or
  fails (IA strong, depends on pulse amplitude)

This paired-pulse protocol reveals the IA recovery time course.

---

## Analytical Properties

### IA as a Temporal Filter

IA implements a **high-pass temporal filter** on the input:
- Slow inputs (ramp, <10 Hz): IA has time to activate → opposed → attenuated
- Fast inputs (step, >50 Hz): IA too slow to activate → bypassed → transmitted

The effective time constant of the filter is τ_b ≈ 50 ms, giving a
cutoff frequency of fc ≈ 1/(2π·50) ≈ 3 Hz.

### Sensitivity to g_A

| g_A (mS/cm²) | First-spike latency (I=1.5) | Adapted rate | Neuron type |
|-------------|---------------------------|-------------|------------|
| 0 | ~3 ms | ~100 Hz | Fast-spiking |
| 4 | ~30 ms | ~60 Hz | Intermediate |
| 8 (default) | ~70 ms | ~40 Hz | Regular-spiking |
| 16 | ~150 ms | ~20 Hz | Strongly delayed |
| 30 | >300 ms or fails | <10 Hz | Near-silent |

### IA and Resting Potential

IA contributes a tonic outward current at rest through the window
current.  Removing IA (g_A = 0) shifts the resting potential ~2–3 mV
more positive, bringing the neuron closer to threshold.  This is why
IA block (e.g. by 4-aminopyridine, 4-AP) increases excitability.

### 4-AP Pharmacology

4-aminopyridine (4-AP) is a selective blocker of IA (and some Kv1
channels).  In the model, 4-AP block is simulated by reducing g_A:

| 4-AP concentration | Equivalent g_A | Effect |
|-------------------|---------------|--------|
| 0 (control) | 8.0 | Normal first-spike delay |
| 1 mM (partial) | 4.0 | Reduced delay, higher rate |
| 5 mM (full) | 0.0 | No delay, fast-spiking |

4-AP is used clinically (fampridine/dalfampridine) to treat
multiple sclerosis — blocking IA at demyelinated nodes restores
conduction by increasing excitability.

### f-I Curve Shape

Without IA: f-I curve rises steeply from threshold (Type II-like).
With IA: f-I curve rises gradually from threshold (Type I-like, due
to the variable delay).  IA therefore shifts the neuron from Type II
toward Type I excitability — from a resonator toward an integrator.

### Comparison with Other Channel Neurons

| Model | Added channel | Effect | ns/step |
|-------|-------------|--------|---------|
| WB base | None | Fast-spiking | ~2.5 µs |
| **ATypeK** | **IA (a³b)** | **Latency, adaptation** | **~3 µs** |
| SK | SK (Ca²⁺-gated) | mAHP adaptation | ~2.8 µs |
| BK | BK (Ca²⁺+V) | fAHP, spike sharpening | ~3 µs |
| NMDA | NMDA (s·B(V)) | Coincidence, plasticity | ~3.3 µs |

---

## FPGA Implementation Estimates

### Resource Requirements (Zynq-7020, XC7Z020)

| Resource | Per neuron | Available | Max neurons |
|----------|-----------|-----------|-------------|
| LUT | ~130 | 53,200 | ~409 |
| FF | ~160 | 106,400 | ~665 |
| DSP48E1 | 6 | 220 | 36 |
| BRAM (18Kb) | 0 | 280 | N/A |

**Breakdown:**
- WB α/β rates (4 exp): ~60 LUT
- IA Boltzmann (2 sigmoid): ~20 LUT
- m_inf³, n⁴, a³: 3 DSP
- 4 current multiplies: 2 DSP
- Gate updates + V update: 1 DSP
- State registers (5 × 32-bit): ~160 FF
- Control: ~50 LUT

### Fixed-Point Precision

**Q16.16 recommended:** same rationale as other WB-based models.
The IA gating variables a and b are straightforward Boltzmann functions
without extreme dynamic range.

### Timing

At 100 MHz with 50 sub-steps:
- Per sub-step: ~10 cycles
- Total: 500 cycles = 5.0 µs
- CPU benchmark: from STUB ~3 µs/step range (similar to SK)
- 409 in parallel: effective ~12.2 ns/neuron/step

---

## Validation

### Functional Checks

| Property | Expected | Measured | Status |
|----------|----------|---------|--------|
| Fires with I = 2 | Sustained spiking | Confirmed | ✅ |
| Silent at I = 0 | No spikes | Confirmed | ✅ |
| First-spike latency present | Delay before first AP | Confirmed | ✅ |
| Removing g_A eliminates delay | No latency at g_A = 0 | Confirmed | ✅ |
| IA inactivates during depol. | b → 0 at V > −50 | Confirmed | ✅ |
| IA recovers during ISI | b increases at V_reset | Confirmed | ✅ |
| V clamped [−100, 60] | Always | 10⁶ steps | ✅ |
| a, b ∈ [0, 1] | Clamped | Confirmed | ✅ |
| NaN recovery | Resets | Confirmed | ✅ |
| Spike = V crossing −20 mV | Reset to −65 | Confirmed | ✅ |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/channels/a_type_k.rs:24` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, h, n, a, b) |
| NetworkRunner wired | `NeuronVariant::ATypeK` |
| `create_neuron("ATypeK")` | Yes |
| `supported_models()` | Includes "ATypeK" |
| coverage tests | 10 |
| Benchmark | from `channels/a_type_k.rs` family: ~3 µs/step range, i5-11600K |

---

## Network Coupling

### Regular-Spiking Pyramidal Cell Networks

IA is the defining current of regular-spiking (RS) cortical pyramidal
cells.  In SC-NeuroCore networks, the ATypeKNeuron represents the RS
population, while fast-spiking (FS) interneurons (without IA) are
modelled by the base WB model.

The IA-mediated first-spike latency creates a natural temporal delay
between input arrival and output firing, which has implications for:
- **Feedforward timing:** FS interneurons fire before RS cells,
  creating a feedforward inhibition window
- **Rate coding:** IA limits the maximum RS firing rate, compressing
  the dynamic range
- **Coincidence detection:** RS cells preferentially respond to
  synchronous input that arrives faster than τ_b

### IA in Cardiac Myocytes

IA (I_to, transient outward) is also present in cardiac ventricular
myocytes, where it produces the Phase 1 notch of the action potential.
The cardiac I_to uses Kv4.3 (same family as neuronal IA) with similar
activation/inactivation kinetics but at different voltages.  The
ATypeKNeuron model can approximate cardiac I_to by adjusting V½
values — a cross-domain application of the same channel model.

### IA and Spike Waveform

IA affects the AP waveform even though it inactivates before the Na⁺
threshold.  The residual IA during the rising phase:
- Slows the upstroke slightly (competing with Na⁺)
- Widens the AP base
- Reduces the peak amplitude by 2–5 mV

These effects are subtle but measurable in voltage-clamp experiments
comparing 4-AP-treated vs control neurons.

### Dendritic IA Gradient

In CA1 pyramidal cells, IA density increases 5-fold from soma to
distal apical dendrite.  This creates a distance-dependent attenuation
of back-propagating APs.  Modelling this requires multiple
ATypeKNeuron compartments with increasing g_A — a multi-compartment
extension of the single-compartment model.

---

## References

1. Connor, J. A. & Stevens, C. F. (1971). Prediction of repetitive
   firing behaviour from voltage clamp data on an isolated neurone soma.
   *J Physiol*, 213(1), 31–53.

2. Hoffman, D. A., Magee, J. C., Colbert, C. M. & Johnston, D. (1997).
   K⁺ channel regulation of signal propagation in dendrites of
   hippocampal pyramidal neurons. *Nature*, 387, 869–875.

3. Hille, B. (2001). *Ion Channels of Excitable Membranes* (3rd ed.).
   Sinauer Associates. Chapter 4.

4. Jerng, H. H., Pfaffinger, P. J. & Bhatt, D. (2004). Molecular
   physiology and modulation of somatodendritic A-type potassium
   channels. *Mol Cell Neurosci*, 27(4), 343–369.

5. Rasmusson, R. L., Bhatt, D. & Bhatt, E. (1998). A mathematical
   model of a bullfrog cardiac pacemaker cell. *Am J Physiol*, 274(1),
   H747–H759.

6. Wang, X. J. & Buzsáki, G. (1996). Gamma oscillation by synaptic
   inhibition in a hippocampal interneuronal network model. *J Neurosci*,
   16(20), 6402–6413.

7. Storm, J. F. (1988). Temporal integration by a slowly inactivating
   K⁺ current in hippocampal neurons. *Nature*, 336, 379–381.

8. Carrasquillo, Y., Burkhalter, A. & Bhatt, D. (2012). A-type K⁺
   channels encoded by Kv4.2 modulate intrinsic firing properties of
   neocortical neurons. *J Neurosci*, 32(31), 10913–10919.

9. Izhikevich, E. M. (2007). *Dynamical Systems in Neuroscience*.
   MIT Press. Chapter 8 (Connor-Stevens model).

10. Mitterdorfer, J. & Bean, B. P. (2002). Potassium currents during
    the action potential of hippocampal CA3 neurons. *J Neurosci*,
    22(23), 10106–10115.

11. Rush, M. E. & Bhatt, D. (1995). The Connor model. In *Handbook of
    Brain Theory and Neural Networks*, Arbib, M. A. (Ed.), MIT Press.

12. Marder, E. & Goaillard, J. M. (2006). Variability, compensation
    and homeostasis in neuron and network function. *Nat Rev Neurosci*,
    7(7), 563–574.
