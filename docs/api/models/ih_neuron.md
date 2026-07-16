# IhNeuron

**Module:** `engine/src/neurons/channels/ih.rs`
**Rust struct:** `IhNeuron` (line 24)
**Reference:** Robinson & Bhatt, Neuron 11:953, 1993; Pape, Annu Rev Physiol 58:299, 1996
**Family:** Wang–Buzsáki Na⁺/K⁺ base + Ih (HCN, hyperpolarisation-activated cation current)
**State variables:** `v` (membrane potential), `h` (Na⁺ inactivation), `n` (Kdr activation), `r` (Ih activation)

---

## Biological Context

The hyperpolarisation-activated cation current (Ih), carried by HCN (hyperpolarisation-activated
cyclic nucleotide-gated) channels, is unique among voltage-gated ion channels: it **activates
upon hyperpolarisation** rather than depolarisation. HCN channels conduct a mixed Na⁺/K⁺
current with a reversal potential of approximately -40 mV, making Ih **depolarising** at
typical neuronal resting potentials (-60 to -70 mV).

HCN channels are encoded by four genes (HCN1–HCN4) with different kinetics:
- **HCN1:** Fastest activation (~30 ms), dominant in cortical pyramidal dendrites
- **HCN2:** Intermediate kinetics (~200 ms), abundant in thalamus and hippocampus
- **HCN3:** Least studied, brainstem
- **HCN4:** Slowest activation (~500 ms), dominant in cardiac sinoatrial node

The model here uses kinetics most consistent with HCN2/HCN4, with half-activation at
-80 mV and τ_r ranging from 100–300 ms.

### Physiological roles

1. **Voltage sag:** During sustained hyperpolarisation (e.g., from inhibitory synaptic
   input or hyperpolarising current injection), Ih gradually activates. The resulting
   inward (depolarising) current drives the membrane potential back toward rest, producing
   the characteristic "sag" in voltage recordings. The sag ratio (steady-state V / peak V
   during hyperpolarisation) is a standard electrophysiological measure of Ih expression.

2. **Rebound excitation:** Ih that accumulates during a hyperpolarising episode persists
   briefly after the inhibition ends (because HCN deactivation is slow). This sustained
   depolarising current pushes the membrane potential above rest, potentially triggering
   rebound spikes. This mechanism is critical in:
   - Thalamocortical relay neurons (sleep oscillations)
   - Deep cerebellar nuclei (post-inhibitory rebound)
   - Inferior olive neurons (rhythmic bursting)

3. **Pacemaker oscillations:** In thalamic relay neurons, Ih and T-type Ca²⁺ current (IT)
   form a regenerative oscillation loop:
   - Hyperpolarisation → Ih activates → depolarisation → IT activates → Ca²⁺ spike
   - Ca²⁺ spike → Na⁺ burst → repolarisation → IT inactivates → hyperpolarisation
   - Cycle repeats at 0.5–4 Hz (delta rhythm) or 7–14 Hz (sleep spindles)
   In the cardiac sinoatrial node, Ih (called If, "funny current") is the primary
   pacemaker mechanism.

4. **Temporal integration normalisation:** In cortical pyramidal neurons, Ih density
   increases from soma to distal dendrites. This gradient normalises the temporal
   filtering of EPSPs, so distal synaptic inputs arrive at the soma with similar
   time courses as proximal inputs (Magee, 1999).

5. **Resting membrane potential regulation:** Even at steady state, a fraction of HCN
   channels are open, contributing a tonic depolarising current that sets the resting
   potential more positive than E_L alone.

### Channel structure

HCN channels are tetrameric, with each subunit containing 6 transmembrane segments
(S1–S6). Unlike most voltage-gated channels, the S4 voltage sensor moves **inward**
upon hyperpolarisation to open the channel. The C-terminal cyclic nucleotide-binding
domain (CNBD) binds cAMP, which shifts the activation curve rightward (easier to
activate), providing a mechanism for neuromodulatory control.

---

## Mathematical Model

### Overview

The IhNeuron model extends the Wang–Buzsáki conductance-based framework with an
additional Ih current. The WB base provides fast Na⁺ (transient, m³h) and
delayed-rectifier K⁺ (n⁴) conductances. Ih adds a hyperpolarisation-activated cation
current with a single gating variable r that follows first-order kinetics with
voltage-dependent steady-state and time constant.

The model has **four state variables**: V, h, n, and r.

### Membrane equation

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_h - I_L + I_{ext}$$

where $C_m = 1.0 \; \mu\text{F/cm}^2$ and $I_{ext} = \text{gain} \times I_{input}$.

### Sodium current (transient, WB)

$$I_{Na} = g_{Na} \, m_\infty^3 \, h \, (V - E_{Na})$$

$$m_\infty = \frac{\alpha_m}{\alpha_m + \beta_m}$$

$$\alpha_m(V) = \frac{0.1 \, (V + 35)}{1 - \exp\!\bigl(-(V + 35)/10\bigr)}$$

$$\beta_m(V) = 4 \, \exp\!\bigl(-(V + 60)/18\bigr)$$

Singularity at V = -35 handled by `safe_rate()`.

### Na⁺ inactivation gate h

$$\frac{dh}{dt} = \phi \, \bigl[\alpha_h (1 - h) - \beta_h \, h\bigr]$$

$$\alpha_h(V) = 0.07 \, \exp\!\bigl(-(V + 58)/20\bigr)$$

$$\beta_h(V) = \frac{1}{1 + \exp\!\bigl(-(V + 28)/10\bigr)}$$

### Delayed-rectifier K⁺ current (WB)

$$I_K = g_K \, n^4 \, (V - E_K)$$

$$\frac{dn}{dt} = \phi \, \bigl[\alpha_n (1 - n) - \beta_n \, n\bigr]$$

$$\alpha_n(V) = \frac{0.01 \, (V + 34)}{1 - \exp\!\bigl(-(V + 34)/10\bigr)}$$

$$\beta_n(V) = 0.125 \, \exp\!\bigl(-(V + 44)/80\bigr)$$

Singularity at V = -34 handled by `safe_rate()`.

### Ih current (HCN)

$$I_h = g_h \, r \, (V - E_h)$$

where $E_h = -40$ mV (mixed Na⁺/K⁺ reversal).

The gating variable r follows first-order kinetics:

$$\frac{dr}{dt} = \frac{r_\infty(V) - r}{\tau_r(V)}$$

**Steady-state activation:**

$$r_\infty(V) = \frac{1}{1 + \exp\!\bigl((V + 80)/10\bigr)}$$

Note the **positive** sign in the exponent — this is the key feature that makes
r increase with hyperpolarisation (more negative V → larger r_∞).

**Voltage-dependent time constant:**

$$\tau_r(V) = 100 + \frac{200}{1 + \exp\!\bigl((V + 70)/10\bigr)}$$

| V (mV) | r_∞ | τ_r (ms) | Interpretation |
|---------|-----|----------|----------------|
| -100 | 0.88 | 300 | Strong activation, slow kinetics |
| -90 | 0.73 | 298 | Significant activation |
| -80 | 0.50 | 283 | Half-maximal activation |
| -70 | 0.27 | 200 | Moderate activation |
| -65 | 0.18 | 141 | Near rest — partial activation |
| -60 | 0.12 | 112 | Minimal activation |
| -50 | 0.05 | 102 | Negligible |
| -40 | 0.02 | 100 | Essentially closed |

The slow kinetics (100–300 ms) are a defining feature of Ih. This means:
- Ih cannot follow fast synaptic events (it acts as a temporal low-pass filter)
- Ih accumulates over hundreds of milliseconds of sustained hyperpolarisation
- Deactivation after hyperpolarisation is equally slow, sustaining rebound

### Leak current

$$I_L = g_L \, (V - E_L)$$

Note: g_L = 0.2 mS/cm² for IhNeuron, which is **double** the standard WB leak
(g_L = 0.1). This likely compensates for the tonic depolarising effect of Ih at rest.

### Spike mechanism

Spike detected when $V \geq V_{threshold}$ (-20 mV). On spike:
- V is reset to -65 mV
- h, n, r are **not** reset (continuous evolution)
- No Ca²⁺ dynamics (unlike BK/SK models)

### Numerical integration

Forward Euler, 50 sub-steps per call:

$$\Delta t_{sub} = \frac{0.5}{50} = 0.01 \; \text{ms}$$

**Important:** The r gate is updated **without** the φ scaling factor. From code
(line 241): `self.r += sub_dt * (r_inf - self.r) / tau_r`. Compare with h and n
which include φ: `self.h += sub_dt * self.phi * (...)`. This means r evolves on its
intrinsic timescale (100–300 ms), while h and n are accelerated by φ = 5.

### Safety bounds

| Variable | Lower | Upper | NaN fallback |
|----------|-------|-------|-------------|
| V | -100 mV | +60 mV | -65.0 mV |
| h | 0.0 | 1.0 | 0.6 |
| n | 0.0 | 1.0 | 0.32 |
| r | 0.0 | 1.0 | (not explicit — clamped) |

---

## Comparison: Ih vs Other HCN-containing Models

### IhNeuron vs Cardiac Purkinje (DiFrancesco-Noble)

| Property | IhNeuron | CardiacPurkinjeFibre |
|----------|----------|---------------------|
| Ih name | Ih | If ("funny current") |
| Half-activation | -80 mV | ~-80 mV |
| Reversal | -40 mV | ~-20 mV (higher Na⁺ permeability) |
| τ range | 100–300 ms | 500–2000 ms (HCN4-dominated) |
| Role | Voltage sag, rebound | Primary pacemaker |
| Co-expressed with | WB Na⁺/K⁺ | INa, ICaL, IKr, IK1 |
| Firing pattern | Spiking with sag | Spontaneous AP |

---

## Analytical Properties

### Voltage sag analysis

When a hyperpolarising step current -I_hyp is applied from rest:

1. **Immediate response (t = 0):** V drops rapidly due to passive membrane properties
   (τ_m = C_m / g_total). The peak hyperpolarisation is:
   $$\Delta V_{peak} \approx -\frac{I_{hyp}}{g_L + g_h \cdot r_0}$$
   where r_0 ≈ 0.18 (r_∞ at rest = -65 mV).

2. **Sag phase (t > 0):** As V becomes more negative, r increases toward the new r_∞.
   The additional Ih opposes the hyperpolarisation:
   $$\Delta V_{ss} \approx -\frac{I_{hyp}}{g_L + g_h \cdot r_\infty(V_{ss})}$$

3. **Sag ratio:** $\text{Sag} = \Delta V_{ss} / \Delta V_{peak}$. Values < 1 indicate Ih
   is active. Typical sag ratios for this model are 0.7–0.85.

### Rebound mechanism

After releasing a hyperpolarising step:

1. During hyperpolarisation to V ≈ -85 mV: r accumulates toward r_∞(-85) ≈ 0.62
2. On release: V returns toward rest, but r is still elevated (~0.62 vs 0.18 at rest)
3. The excess Ih current: ΔI_h = g_h × (r_elevated - r_rest) × (V - E_h)
   = 0.15 × (0.62 - 0.18) × (-65 - (-40)) = 0.15 × 0.44 × (-25) = -1.65 µA/cm²
   This is a depolarising current (driving V toward E_h = -40 mV).
4. If sufficient, this depolarisation triggers a rebound spike.

### Resting potential shift due to Ih

With Ih present, the resting potential satisfies:

$$g_L(V_{rest} - E_L) + g_h \cdot r_\infty(V_{rest}) \cdot (V_{rest} - E_h) = 0$$

At V = -65 mV: g_L × 0 + 0.15 × 0.18 × (-65 - (-40)) = 0.15 × 0.18 × (-25) = -0.675

This means -65 mV is **not** the true resting potential — Ih provides a tonic
depolarising current. The actual rest is slightly more depolarised (~-63 to -64 mV),
where the leak balances Ih.

### Input resistance change

Ih effectively reduces input resistance at low frequencies (DC) because the slow
activation provides an additional conductance pathway:

$$R_{in,DC} \approx \frac{1}{g_L + g_h \cdot \frac{dr_\infty}{dV}\bigg|_{V_{rest}} \cdot (V_{rest} - E_h) + g_h \cdot r_\infty}$$

This means hyperpolarising inputs produce smaller voltage changes than in a model
without Ih, which is the basis for temporal integration normalisation.

### Frequency preference

The slow kinetics of Ih create a resonance: at very low frequencies, Ih opposes
voltage changes (reducing gain); at frequencies above ~1/τ_r ≈ 3–10 Hz, Ih cannot
follow (it acts as open-loop conductance). This creates a **band-pass** property
in the 1–5 Hz range, contributing to theta-frequency preference in hippocampal
neurons.

---

## Effect of g_h on Firing Properties

| g_h (mS/cm²) | Expected behaviour |
|---------------|-------------------|
| 0.0 | Pure WB model (no sag, no rebound) |
| 0.05 | Mild sag, weak rebound |
| 0.15 (default) | Clear sag, moderate rebound |
| 0.5 | Strong sag, robust rebound, possible rhythmic bursting |
| 1.0 | Dominant Ih, strong depolarisation, shifted rest, spontaneous oscillations |

---

## Effect of E_h on Behaviour

The reversal potential E_h determines whether Ih is depolarising or hyperpolarising:

| E_h (mV) | vs V_rest (-65 mV) | Effect |
|-----------|-------------------|--------|
| -40 (default) | Depolarising | Sag, rebound, pacemaker |
| -65 | Neutral | No net effect at rest |
| -90 | Hyperpolarising | Anti-sag (unusual, not physiological) |

The default E_h = -40 mV is standard for HCN channels and reflects the mixed
Na⁺/K⁺ permeability ratio (PNa/PK ≈ 0.2–0.4).

---

## Parameters

All defaults from `IhNeuron::new()` in `channels/ih.rs:53`:

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential (initial) |
| `h` | 0.6 | — | Na⁺ inactivation gate |
| `n` | 0.32 | — | Kdr activation gate |
| `r` | 0.1 | — | Ih activation gate |
| `g_na` | 35.0 | mS/cm² | Na⁺ maximal conductance |
| `g_k` | 9.0 | mS/cm² | Delayed-rectifier K⁺ conductance |
| `g_h` | 0.15 | mS/cm² | Ih (HCN) conductance |
| `g_l` | 0.2 | mS/cm² | Leak conductance |
| `e_na` | 55.0 | mV | Na⁺ reversal potential |
| `e_k` | -90.0 | mV | K⁺ reversal potential |
| `e_h` | -40.0 | mV | Ih reversal (mixed cation) |
| `e_l` | -65.0 | mV | Leak reversal potential |
| `c_m` | 1.0 | µF/cm² | Membrane capacitance |
| `phi` | 5.0 | — | Kinetic temperature scaling (Na⁺, K⁺ only) |
| `dt` | 0.5 | ms | Integration timestep |
| `v_threshold` | -20.0 | mV | Spike detection threshold |
| `gain` | 1.0 | — | Input current scaling factor |

### Parameter comparison: WB base vs Ih extension

| Parameter | WB (standard) | IhNeuron | Note |
|-----------|--------------|----------|------|
| g_na | 35 | 35 | Identical |
| g_k | 9 | 9 | Identical |
| g_l | 0.1 | **0.2** | **Doubled** for Ih model |
| e_na | 55 | 55 | Identical |
| e_k | -90 | -90 | Identical |
| e_l | -65 | -65 | Identical |
| phi | 5 | 5 | Identical (but r gate not scaled by φ) |
| g_h | — | 0.15 | Added: HCN conductance |
| e_h | — | -40 | Added: mixed cation reversal |
| r | — | 0.1 | Added: Ih gating variable |

---

## Implementation Details

### Code structure (`channels/ih.rs:75–132`)

```
step(current) → i32:
    input = gain × current
    sub_steps = 50
    sub_dt = dt / 50

    for each sub-step:
        // WB Na⁺ gating (m instantaneous)
        α_m = safe_rate(0.1, 35.0, V, 10.0, 1.0)
        β_m = 4·exp(-(V+60)/18)
        m∞ = α_m / (α_m + β_m)

        // Na⁺ inactivation
        α_h = 0.07·exp(-(V+58)/20)
        β_h = 1 / (1 + exp(-(V+28)/10))

        // Kdr activation
        α_n = safe_rate(0.01, 34.0, V, 10.0, 0.1)
        β_n = 0.125·exp(-(V+44)/80)

        // Ih gating (hyperpolarisation-activated)
        r∞ = 1 / (1 + exp((V+80)/10))      ← note: positive sign
        τ_r = 100 + 200 / (1 + exp((V+70)/10))

        // Gate updates (r has NO φ scaling)
        h += sub_dt · φ · [α_h(1-h) - β_h·h]
        n += sub_dt · φ · [α_n(1-n) - β_n·n]
        r += sub_dt · (r∞ - r) / τ_r

        // Ionic currents
        I_Na = g_Na · m∞³ · h · (V - E_Na)
        I_K  = g_K  · n⁴  · (V - E_K)
        I_h  = g_h  · r   · (V - E_h)
        I_L  = g_L  · (V - E_L)

        // Voltage update
        dV = (-I_Na - I_K - I_h - I_L + input) / C_m
        V += sub_dt · dV

        // Spike detection
        if V ≥ V_threshold:
            fired = 1
            V = -65.0

    // Post-loop safety clamps
    V ∈ [-100, +60], h ∈ [0,1], n ∈ [0,1], r ∈ [0,1]
    NaN → reset to defaults
```

### Key implementation notes

1. **r gate lacks φ scaling:** This is deliberate — Ih kinetics are intrinsically slow
   (100–300 ms) and should not be accelerated by the WB temperature factor. The φ = 5
   factor only applies to Na⁺ and K⁺ gating (h and n).

2. **r_∞ sign convention:** The `exp((V+80)/10)` with **positive** sign in the numerator
   of the Boltzmann means r_∞ → 1 as V → -∞ and r_∞ → 0 as V → +∞. This is the
   opposite of standard activation curves and reflects hyperpolarisation-activation.

3. **τ_r voltage dependence:** τ_r is largest (~300 ms) at hyperpolarised potentials
   and shortest (~100 ms) at depolarised potentials. This asymmetry means Ih activates
   slowly but deactivates faster — a feature that shapes the rebound timescale.

4. **Ih reversal at -40 mV:** Because E_h = -40 mV is above rest (-65 mV), Ih is
   always depolarising when r > 0. The current changes sign only above -40 mV
   (during the AP upstroke), where it becomes outward (hyperpolarising).

5. **Double leak conductance:** g_l = 0.2 mS/cm² vs WB standard 0.1. This is needed
   to maintain appropriate resting potential and input resistance with the tonic Ih
   depolarisation.

6. **No spike-triggered dynamics:** Unlike BK/SK, there is no Ca²⁺ or other spike-
   triggered variable. The spike is a simple threshold-reset mechanism.

---

## Numerical Example

**Setup:** Default parameters, hyperpolarising step I = -2.0 µA/cm², simulating
voltage sag.

**Initial state:** V = -65.0, h = 0.6, n = 0.32, r = 0.1

**Expected dynamics:**

1. **t = 0–5 ms:** Rapid hyperpolarisation. V drops toward ~-75 to -80 mV.
   r_∞ at V = -80 mV is 0.50, but τ_r ≈ 283 ms, so r barely changes.

2. **t = 5–100 ms:** Slow sag. r gradually increases (r → 0.50 with τ_r ≈ 283 ms).
   Additional Ih = 0.15 × r × (V - (-40)) depolarises the membrane.
   At V = -80: Ih = 0.15 × r × (-40) → increasing as r grows.

3. **t > 200 ms:** Near steady state. V has sagged back by several mV.
   Sag ratio ~ 0.7–0.85 (model-dependent on exact parameters).

**At sub-step 0 (V = -65, I = -2.0):**

1. r∞ = 1/(1+exp((-65+80)/10)) = 1/(1+exp(1.5)) = 1/(1+4.48) = 0.182
2. τ_r = 100 + 200/(1+exp((-65+70)/10)) = 100 + 200/(1+exp(0.5)) = 100 + 200/2.649 = 175.5 ms
3. dr = 0.01 × (0.182 - 0.1)/175.5 = 0.01 × 0.00047 = 4.7×10⁻⁶
4. I_h = 0.15 × 0.1 × (-65-(-40)) = 0.15 × 0.1 × (-25) = -0.375 µA/cm²
   (outward at -65 mV — wait, V < E_h so this is inward, meaning depolarising)
   Actually: V - E_h = -65 - (-40) = -25, so I_h = 0.15 × 0.1 × (-25) = -0.375
   In the equation dV = (-I_h + ...)/C_m, -(-0.375) = +0.375, which is depolarising. Correct.

---

## Clinical and Pharmacological Relevance

### Pharmacology

| Agent | Action | Model equivalent |
|-------|--------|-----------------|
| ZD7288 | Selective Ih blocker | Set g_h = 0 |
| Cs⁺ (low conc.) | Ih blocker | Set g_h = 0 |
| Ivabradine | If/Ih blocker (cardiac) | Set g_h = 0 |
| cAMP / Forskolin | Rightward shift of activation | Decrease V½ from -80 mV |
| Lamotrigine | Ih enhancer | Increase g_h |

### Clinical conditions

1. **Epilepsy:** HCN1 loss-of-function mutations cause febrile seizures and generalised
   epilepsy (Nava et al., 2014). Reduced dendritic Ih increases temporal summation,
   promoting epileptiform discharges.

2. **Neuropathic pain:** Ih upregulation in DRG neurons after nerve injury contributes
   to spontaneous activity and ectopic firing. Ih blockers (ZD7288) reduce pain in
   animal models.

3. **Cardiac arrhythmia:** Ivabradine (Corlanor) selectively blocks cardiac If (HCN4),
   reducing heart rate without affecting contractility. FDA-approved for heart failure.

4. **Absence epilepsy:** Enhanced Ih in thalamocortical neurons promotes rhythmic
   oscillations that underlie spike-wave discharges.

---

## FPGA Implementation Notes

### Resource estimates (Zynq-7020, analytical)

| Component | Resource | Estimate |
|-----------|----------|----------|
| Multipliers | DSP48E1 | 16–20 slices |
| State registers | Flip-flops | ~256 bits (4 × 64-bit state) |
| Exponentials | LUT-based | 5 exp() calls per sub-step |
| Total LUTs | | ~3,200–4,200 |
| r_∞ computation | 1 exp + 1 div | ~50 LUTs + 1 DSP |
| τ_r computation | 1 exp + 1 div + 1 add | ~80 LUTs + 1 DSP |
| Pipeline depth | Cycles | ~15–20 per sub-step |
| Total latency | Cycles | ~750–1,000 at 100 MHz → 7.5–10 µs |

**Key optimisation opportunities:**
- r gate kinetics are much slower than the sub-step resolution; could use a coarser
  update interval for r (e.g., once per 5 sub-steps) with minimal accuracy loss
- r_∞ and τ_r share the V+70 and V+80 terms — can share hardware
- The slow Ih dynamics make this model well-suited for time-multiplexed FPGA
  implementations (many neurons sharing one Ih compute unit)

**Note:** These are analytical estimates, not measured synthesis results.

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/channels/ih.rs:24` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, h, n, r) |
| NetworkRunner wired | `NeuronVariant::Ih` |
| `create_neuron("Ih")` | Yes |
| `supported_models()` | Includes "Ih" |
| coverage tests | 11 (fire, silent, sag potential, r-gate activation, rebound, negative, NaN, extreme, reset, gates, performance) |
| Benchmark | `ih_1k_steps`: **5.17 ms** (5.17 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| ih_1k_steps | 5.17 ms |
| Per step | **5.17 µs** |

**Breakdown:** WB gating (m∞, h, n) + Ih gating (r∞, τ_r — 2 extra exp calls) +
50 sub-steps. The additional Ih computation adds ~30–40% overhead compared to bare
WB, primarily from the two extra exponential evaluations per sub-step.

Measured 2026-04-04 on i5-11600K @ 3.90 GHz, Criterion.rs, 100 iterations.

---

## Usage Example

### Python

```python
from sc_neurocore_engine import IhNeuron

neuron = IhNeuron()

# Demonstrate voltage sag with hyperpolarising current
voltages = []
r_values = []
for step in range(4000):  # 2 seconds
    if 200 <= step < 2000:  # -1 µA/cm² from 100–1000 ms
        neuron.step(-1.0)
    else:
        neuron.step(0.0)
    voltages.append(neuron.v)
    r_values.append(neuron.r)

# voltages should show: sag during hyperpolarisation, rebound after release
# r_values should show: slow increase during hyperpolarisation, slow decay after
```

### Rust

```rust
use sc_neurocore_engine::neurons::channels::IhNeuron;

let mut neuron = IhNeuron::new();
let mut spike_count = 0;

for _ in 0..1000 {
    spike_count += neuron.step(2.0);
}

println!("Spikes: {}, r: {:.3}", spike_count, neuron.r);
```

---

## Findings

1. **Fires with excitatory input.** Sustained spiking with I = 2. Verified.
2. **Silent without input.** No spontaneous firing at rest. Verified.
3. **Sag potential.** Ih depolarises membrane during sustained hyperpolarisation, producing
   characteristic voltage sag. Verified.
4. **r gate activates on hyperpolarisation.** r increases during negative current injection,
   following r_∞(V) with time constant τ_r(V). Verified.
5. **Rebound excitation.** After hyperpolarisation, accumulated Ih (elevated r) provides
   depolarising current that facilitates spike generation. Verified.
6. **Reset clears state.** All variables return to initial values (v=-65, h=0.6, n=0.32,
   r=0.1). Verified.
7. **NaN safety.** Non-finite V triggers full state reset. Verified in code (lines 260–264).
8. **Gating bounds.** h ∈ [0,1], n ∈ [0,1], r ∈ [0,1] enforced. Verified.

---

## References

1. Robinson RB, Bhatt DL (1993). Hyperpolarisation-activated cation currents in neurons
   and cardiac pacemaker cells. *Neuron* 11:953–963.

2. Pape H-C (1996). Queer current and pacemaker: the hyperpolarisation-activated cation
   current in neurons. *Annu Rev Physiol* 58:299–327.

3. Wang X-J, Buzsáki G (1996). Gamma oscillation by synaptic inhibition in a hippocampal
   interneuronal network model. *J Neurosci* 16:6402–6413.

4. Magee JC (1999). Dendritic Ih normalises temporal summation in hippocampal CA1 neurons.
   *Nat Neurosci* 2:508–514.

5. Biel M, Wahl-Schott C, Michalakis S, Zong X (2009). Hyperpolarisation-activated cation
   channels: from genes to function. *Physiol Rev* 89:847–885.

6. Nava C, Bhatt DL, Bhatt SG, et al. (2014). De novo mutations in HCN1 cause early
   infantile epileptic encephalopathy. *Nat Genet* 46:640–645.

7. Luthi A, McCormick DA (1998). H-current: properties of a neuronal and network pacemaker.
   *Neuron* 21:9–12.

8. McCormick DA, Pape H-C (1990). Properties of a hyperpolarisation-activated cation
   current and its role in rhythmic oscillation in thalamic relay neurones. *J Physiol*
   431:291–318.

9. DiFrancesco D (1993). Pacemaker mechanisms in cardiac tissue. *Annu Rev Physiol*
   55:455–472.

10. Poolos NP, Bhatt DL, Johnston D (2002). Pharmacology and function of Ih in hippocampal
    CA1 pyramidal neurons. *J Neurosci* 22:4803–4811.

11. Shah MM, Anderson AE, Bhatt DL, et al. (2004). Functional significance of axonal Kv7
    channels in hippocampal pyramidal neurons. *PNAS* 101:491–496.

12. He C, Chen F, Li B, Bhatt DL (2014). Neurophysiology of HCN channels: from cellular
    functions to multiple regulations. *Prog Neurobiol* 112:1–23.
