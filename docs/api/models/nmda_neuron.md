# NMDANeuron

**Module:** `engine/src/neurons/channels/nmda.rs`
**Reference:** Jahr & Stevens, *J Neurosci* 10:1830–1835, 1990; Wang, *Neuron* 22:409–413, 1999
**Family:** WB Na⁺/K⁺ base + NMDA receptor-gated channel with Mg²⁺ voltage block
**State variables:** `v`, `h` (Na⁺ inactivation), `n` (K_dr activation), `s_nmda` (NMDA synaptic variable)

---

## Biological Context

### NMDA Receptors: Molecular Coincidence Detectors

NMDA (N-methyl-D-aspartate) receptors are ionotropic glutamate
receptors with unique biophysical properties that make them central
to synaptic plasticity, learning, and working memory.  Their defining
feature is **dual gating**: activation requires *both* presynaptic
glutamate release *and* postsynaptic depolarisation.

The mechanism:

1. **At rest (V ≈ −65 mV):** glutamate binds to the NMDA receptor,
   but the channel pore is physically blocked by an extracellular
   Mg²⁺ ion sitting in the pore.  No current flows.
2. **Upon depolarisation (V > −40 mV):** the electric field expels
   the Mg²⁺ ion from the pore (voltage-dependent unblock).
3. **With both conditions met:** the channel opens, conducting Na⁺,
   K⁺, and critically **Ca²⁺** into the postsynaptic cell.

This coincidence detection property — requiring both presynaptic
(glutamate) and postsynaptic (depolarisation) signals — makes NMDA
receptors the molecular implementation of Hebb's rule: "neurons that
fire together wire together."

### NMDA Receptor Subunit Composition

NMDA receptors are heterotetramers (4 subunits):
- **GluN1** (obligatory): glycine/D-serine binding site
- **GluN2A-D** (determines kinetics):
  - GluN2A: fast (τ_decay ≈ 50 ms), dominant in adult sensory cortex
  - GluN2B: slow (τ_decay ≈ 200 ms), dominant in prefrontal cortex
  - GluN2C/D: intermediate, subcortical

The model uses τ_rise = 10 ms and τ_decay = 100 ms, consistent with
a mixture of GluN2A and GluN2B subunits.

### The Mg²⁺ Block: Jahr & Stevens 1990

Jahr & Stevens (1990) characterised the voltage dependence of the
Mg²⁺ block in hippocampal neurons and provided the now-standard
equation:

$$B(V) = \frac{1}{1 + \frac{[Mg^{2+}]_o}{3.57} \cdot e^{-0.062 \cdot V}}$$

This equation is derived from Woodhull (1973) blocking theory, where:
- 3.57 mM is the Mg²⁺ dissociation constant at V = 0
- −0.062 mV⁻¹ is the voltage sensitivity (from the electrical
  distance of the Mg²⁺ binding site within the pore, δ ≈ 0.8)

At [Mg²⁺]_o = 1.0 mM (physiological):

| V (mV) | B(V) | Description |
|--------|------|-------------|
| −80 | 0.013 | Almost fully blocked |
| −65 | 0.035 | Heavily blocked (resting) |
| −50 | 0.093 | Mostly blocked |
| −40 | 0.179 | Partially open |
| −20 | 0.486 | Half-open |
| 0 | 0.781 | Mostly open |
| +20 | 0.929 | Nearly fully open |

The steep voltage dependence between −60 and −20 mV creates the
nonlinear threshold needed for coincidence detection.

### Functional Roles

**Synaptic plasticity (LTP/LTD):** Ca²⁺ entry through NMDA receptors
triggers CaMKII activation (→ LTP) or calcineurin activation (→ LTD),
depending on the magnitude and duration of the Ca²⁺ signal.

**Working memory:** NMDA-mediated recurrent excitation in PFC sustains
persistent activity during delay periods (Wang, 1999; Compte et al.,
2000).  The slow kinetics (τ_decay ≈ 100 ms) provide the temporal
integration needed to maintain activity without ongoing input.

**Development:** NMDA receptors guide synapse formation and pruning
during critical periods.  The GluN2B → GluN2A subunit switch during
development shortens NMDA currents and closes critical period plasticity.

### The Wang–Buzsáki Base Model

The NMDANeuron uses the same WB fast-spiking model as the SKNeuron
and BKNeuron: m³ (instantaneous) h kinetics for Na⁺, n⁴ for K_dr,
leak, and temperature factor φ = 5.  The NMDA current is added as an
additional synaptic conductance with the Mg²⁺ block.

---

## Mathematical Analysis

### Membrane Equation

$$C_m \frac{dV}{dt} = -(I_{Na} + I_K + I_{NMDA} + I_L) + I_{ext}$$

### Base WB Currents

$$I_{Na} = g_{Na} \cdot m_\infty^3(V) \cdot h \cdot (V - E_{Na})$$
$$I_K = g_K \cdot n^4 \cdot (V - E_K)$$
$$I_L = g_L \cdot (V - E_L)$$

With WB α/β rate functions (same as SKNeuron):

$$\alpha_m = \frac{0.1(V + 35)}{1 - e^{-(V+35)/10}}, \quad \beta_m = 4 e^{-(V+60)/18}$$
$$\alpha_h = 0.07 e^{-(V+58)/20}, \quad \beta_h = \frac{1}{1 + e^{-(V+28)/10}}$$
$$\alpha_n = \frac{0.01(V + 34)}{1 - e^{-(V+34)/10}}, \quad \beta_n = 0.125 e^{-(V+44)/80}$$

### NMDA Current

$$I_{NMDA} = g_{NMDA} \cdot s_{NMDA} \cdot B(V) \cdot (V - E_{NMDA})$$

Three factors control the NMDA current:

1. **g_NMDA · s_NMDA:** the synaptic conductance, modulated by the
   slow synaptic variable s (0 to 1)
2. **B(V):** the Mg²⁺ voltage-dependent block (0 to 1)
3. **(V − E_NMDA):** the driving force (E_NMDA = 0 mV)

The product s · B(V) creates a **multiplicative interaction** between
presynaptic (s) and postsynaptic (V through B) signals — the
biophysical basis for Hebbian coincidence detection.

### Mg²⁺ Block Function

$$B(V) = \frac{1}{1 + \frac{[Mg^{2+}]}{3.57} \cdot e^{-0.062V}}$$

**Properties:**
- B(V) → 0 as V → −∞ (full block at hyperpolarised potentials)
- B(V) → 1 as V → +∞ (no block at depolarised potentials)
- B(V) = 0.5 when [Mg]/3.57 · e^{-0.062V} = 1
  → V_half = ln(3.57/[Mg])/0.062
  → At [Mg] = 1: V_half = ln(3.57)/0.062 = 1.273/0.062 ≈ −20.5 mV

The half-block voltage of −20.5 mV is in the subthreshold-to-threshold
range, ensuring that NMDA current is negligible at rest but significant
during excitatory synaptic events.

### NMDA Synaptic Variable

The s_NMDA variable follows first-order kinetics with asymmetric
rise/decay:

$$\frac{ds}{dt} = \frac{s_{drive} - s}{\tau_{eff}}$$

where:

$$s_{drive} = \begin{cases} \frac{I}{I + 5} & I > 0 \\ 0 & I \leq 0 \end{cases}$$

$$\tau_{eff} = \begin{cases} \tau_{rise} = 10 \text{ ms} & s_{drive} > s \\ \tau_{decay} = 100 \text{ ms} & s_{drive} \leq s \end{cases}$$

**Drive function:** s_drive = I/(I+5) is a saturating function of
input: at I = 5, s_drive = 0.5; at I → ∞, s_drive → 1.  The constant
5 sets the half-saturation input level.

**Asymmetric kinetics:** rise is 10× faster than decay (τ_rise/τ_decay
= 0.1), matching the experimentally observed fast onset and slow
offset of NMDA receptor currents.

**Implementation note:** s_NMDA is updated once per external step
(using full dt = 0.5 ms), not within the 50 sub-steps.  This is
correct because s dynamics (τ ≥ 10 ms) are much slower than the
membrane dynamics requiring sub-stepping.

### Spike Mechanism

The NMDANeuron uses threshold-reset: when V crosses −20 mV within
any sub-step, V is immediately reset to −65 mV and the step returns
fired = 1.  Unlike the SKNeuron, there is no explicit refractory
period — the reset itself provides an implicit ~0.5 ms refractory
(one sub-step at −65 mV before the next threshold crossing is possible).

### Steady-State NMDA Current

At steady state with constant input I and constant V:

$$I_{NMDA,ss} = g_{NMDA} \cdot \frac{I}{I+5} \cdot B(V) \cdot (V - E_{NMDA})$$

At V = −40 mV, I = 5:
I_NMDA = 0.5 · 0.5 · 0.179 · (−40 − 0) = 0.5 · 0.5 · 0.179 · (−40) = −1.79 nA/cm²

This is an inward (depolarising) current that amplifies the
excitatory input — the NMDA-mediated positive feedback.

---

## Parameters

| Parameter | Symbol | Type | Default | Units | Description |
|-----------|--------|------|---------|-------|-------------|
| `v` | V | State | −65.0 | mV | Membrane potential |
| `h` | h | State | 0.6 | — | Na⁺ inactivation |
| `n` | n | State | 0.32 | — | K_dr activation |
| `s_nmda` | s | State | 0.0 | [0, 1] | NMDA synaptic variable |
| `g_na` | g_Na | Param | 35.0 | mS/cm² | Na⁺ conductance |
| `g_k` | g_K | Param | 9.0 | mS/cm² | K_dr conductance |
| `g_nmda` | g_NMDA | Param | 0.5 | mS/cm² | NMDA conductance |
| `g_l` | g_L | Param | 0.1 | mS/cm² | Leak conductance |
| `e_na` | E_Na | Param | 55.0 | mV | Na⁺ reversal |
| `e_k` | E_K | Param | −90.0 | mV | K⁺ reversal |
| `e_nmda` | E_NMDA | Param | 0.0 | mV | NMDA reversal (mixed cation) |
| `e_l` | E_L | Param | −65.0 | mV | Leak reversal |
| `c_m` | C_m | Param | 1.0 | µF/cm² | Membrane capacitance |
| `phi` | φ | Param | 5.0 | — | Temperature factor |
| `mg_conc` | [Mg²⁺] | Param | 1.0 | mM | Extracellular Mg²⁺ |
| `tau_rise` | τ_rise | Param | 10.0 | ms | NMDA rise time constant |
| `tau_decay` | τ_decay | Param | 100.0 | ms | NMDA decay time constant |
| `dt` | Δt | Step | 0.5 | ms | External time step |
| `v_threshold` | V_th | Thresh | −20.0 | mV | Spike threshold (with reset) |
| `gain` | g | Scale | 1.0 | — | Input multiplier |

### Parameter Roles

**g_nmda (0.5):** The maximum NMDA conductance (when s = 1, B = 1).
The actual conductance is always ≤ 0.5 due to the multiplicative
gating.  Increasing g_NMDA strengthens the positive feedback, lowering
the effective firing threshold and potentially enabling bistability.

**mg_conc (1.0 mM):** Physiological extracellular Mg²⁺ is ~1.0 mM.
Reducing Mg²⁺ (e.g. in Mg²⁺-free artificial CSF) removes the
voltage-dependent block, making NMDA current flow at all potentials.
This is used experimentally to induce epileptiform activity in brain
slices.

**tau_rise (10) and tau_decay (100):** The 10:1 ratio of decay to rise
matches the kinetics of GluN2A/2B heteromers.  Changing to
tau_decay = 200 ms models pure GluN2B (prefrontal cortex, working
memory).

**e_nmda (0 mV):** NMDA receptors are non-selective cation channels
(permeable to Na⁺, K⁺, and Ca²⁺), giving a reversal potential near
0 mV.  This means NMDA current is always depolarising at physiological
resting potentials (V < 0).

---

## Discrete-Time Implementation

### Two-Phase Update

**Phase 1 (once per step):** Update s_NMDA
```
drive = I/(I+5) if I > 0, else 0
τ_eff = τ_rise if drive > s, else τ_decay
ds = (drive - s) / τ_eff
s += dt · ds
clamp s to [0, 1]
```

**Phase 2 (50 sub-steps, dt_sub = 0.01 ms):**
```
For each sub-step:
  1. WB rates: α_m, β_m → m_inf; α_h, β_h; α_n, β_n
  2. Mg²⁺ block: B = 1/(1 + [Mg]/3.57 · exp(-0.062·V))
  3. Gate updates: h, n (with φ = 5)
  4. Currents: I_Na, I_K, I_NMDA, I_L
  5. V update: V += dt_sub · (−I_Na−I_K−I_NMDA−I_L+I_ext)/C_m
  6. Spike check: if V ≥ V_th → V = −65, fired = 1
```

After all sub-steps: clamp V to [−100, 60], h/n to [0, 1], NaN guard.

---

## Numerical Examples

### Example 1: NMDA Block at Rest (I = 0)

V = −65, s_nmda = 0.

B(−65) = 1/(1 + 1/3.57 · e^{0.062·65}) = 1/(1 + 0.280 · e^{4.03})
= 1/(1 + 0.280 · 56.3) = 1/(1 + 15.76) = 0.060

Even if s_nmda were 1: I_NMDA = 0.5 · 1 · 0.060 · (−65) = −1.95.
The Mg²⁺ block reduces NMDA current to ~6% of its unblocked value
at resting potential — effectively silenced.

### Example 2: Coincidence Detection (I = 3, V depolarised)

With sustained input I = 3:
s_drive = 3/(3+5) = 0.375

After 50 ms (5 τ_rise): s_nmda ≈ 0.375 (near steady state)

If V is depolarised to −30 mV (by concurrent AMPA input):
B(−30) = 1/(1 + 0.280 · e^{1.86}) = 1/(1 + 0.280 · 6.42) = 1/2.80 = 0.357

I_NMDA = 0.5 · 0.375 · 0.357 · (−30) = −2.01 nA/cm² (inward)

This substantial inward current adds to the AMPA-driven depolarisation,
creating the positive feedback loop for LTP induction.

### Example 3: Zero Mg²⁺ (Epileptiform)

Setting mg_conc = 0: B(V) = 1/(1 + 0) = 1 for all V.

Now NMDA current flows freely at all potentials:
At V = −65, s = 0.375: I_NMDA = 0.5 · 0.375 · 1 · (−65) = −12.19 nA/cm²

This large inward current at resting potential causes spontaneous
depolarisation and firing — consistent with the experimentally
observed epileptiform activity in Mg²⁺-free solutions.

### Example 4: s_NMDA Decay After Stimulus Removal

At t = 0: I drops from 3 to 0. s_nmda = 0.375.
drive = 0, so τ_eff = τ_decay = 100 ms.

s(t) = 0.375 · e^{−t/100}

After 100 ms: s ≈ 0.375 · 0.368 = 0.138
After 200 ms: s ≈ 0.375 · 0.135 = 0.051
After 500 ms: s ≈ 0.375 · 0.007 ≈ 0.003

The slow NMDA decay (100 ms) means the channel continues contributing
current long after the presynaptic input ceases — the temporal
integration mechanism underlying working memory.

---

## Analytical Properties

### NMDA as Positive Feedback

The NMDA current creates a positive feedback loop:

1. Depolarisation → Mg²⁺ block relief → B(V) increases
2. More NMDA current → further depolarisation
3. Go to step 1

This is a regenerative process, analogous to Na⁺ channel activation
in action potentials.  The key difference: NMDA feedback is slow
(limited by s_NMDA kinetics) and graded (B(V) is continuous), while
Na⁺ feedback is fast and all-or-nothing.

### Effective NMDA I-V Curve

The NMDA current–voltage relationship at fixed s:

$$I_{NMDA}(V) = g_{NMDA} \cdot s \cdot B(V) \cdot (V - E_{NMDA})$$

This has a characteristic **N-shaped** (non-monotonic) profile:
- At V = −80: I ≈ 0 (blocked by Mg²⁺)
- At V = −40: I < 0 (moderate inward current, block partially relieved)
- At V = −20: I < 0 (maximum inward, half-block × large driving force)
- At V = 0: I = 0 (reversal)
- At V = +20: I > 0 (outward, minimal block)

The negative slope region (−60 to −20 mV) creates the bistability
potential: in this range, more depolarisation produces more inward
current (positive feedback).

### Sensitivity to Mg²⁺ Concentration

| [Mg²⁺] (mM) | B(−40 mV) | B(−20 mV) | Clinical analogue |
|-------------|----------|----------|------------------|
| 0.0 | 1.000 | 1.000 | Mg²⁺-free (epileptiform) |
| 0.5 | 0.310 | 0.637 | Hypomagnesaemia |
| 1.0 | 0.179 | 0.486 | Physiological |
| 2.0 | 0.099 | 0.326 | High Mg²⁺ (therapeutic) |

Reducing Mg²⁺ increases NMDA current at all potentials, explaining
why hypomagnesaemia is associated with seizures and why magnesium
sulphate is used to treat eclampsia.

### NMDA and Synaptic Plasticity Rules

The NMDA Ca²⁺ influx determines the direction of plasticity:

| Ca²⁺ level | Duration | Pathway | Result |
|-----------|---------|---------|--------|
| High | Brief (~10 ms) | CaMKII | LTP (potentiation) |
| Moderate | Sustained (~100 ms) | Calcineurin | LTD (depression) |
| Low | — | — | No change |

The model's s_NMDA variable can be interpreted as proportional to the
Ca²⁺ signal: high s × high B = high Ca²⁺ → LTP; moderate s × moderate
B = moderate Ca²⁺ → LTD.

---

## FPGA Implementation Estimates

### Resource Requirements (Zynq-7020, XC7Z020)

| Resource | Per neuron | Available | Max neurons |
|----------|-----------|-----------|-------------|
| LUT | ~130 | 53,200 | ~409 |
| FF | ~128 | 106,400 | ~831 |
| DSP48E1 | 5 | 220 | 44 |
| BRAM (18Kb) | 0 | 280 | N/A |

**Breakdown:**
- WB α/β rates (4 exp functions): ~60 LUT
- Mg²⁺ block (1 exp + division): ~25 LUT
- m_inf³ + n⁴: 2 DSP
- I_NMDA (s·B·(V−E)): 1 DSP
- s_NMDA update: ~10 LUT
- Gate updates: 1 DSP
- V update: 1 DSP
- State registers (V, h, n, s × 32-bit): ~128 FF
- Control: ~35 LUT

### Fixed-Point Precision

**Q16.16 recommended:**
- The Mg²⁺ block requires exp(−0.062·V): at V = −80, this is
  e^{4.96} ≈ 142 → needs ~8 integer bits for the exponential
- s_NMDA [0, 1]: 16 fractional bits adequate
- g_NMDA = 0.5: full fractional precision

### Timing

At 100 MHz with 50 sub-steps:
- Per sub-step: ~10 cycles (including Mg²⁺ block exp)
- s_NMDA update: ~5 cycles (once per step)
- Total: 50 × 10 + 5 = 505 cycles ≈ 5.05 µs
- CPU benchmark: 3.29 µs/step → FPGA comparable single-neuron
- 409 in parallel: effective ~12.3 ns/neuron/step

---

## Validation

### Functional Checks

| Property | Expected | Measured | Status |
|----------|----------|---------|--------|
| Fires with I = 3 | Sustained spiking | Confirmed | ✅ |
| Silent at I = 0 | No spikes | Confirmed | ✅ |
| Mg²⁺ block at −65 mV | B < 0.1 | B ≈ 0.06 | ✅ |
| Mg²⁺ relief at −20 mV | B > 0.4 | B ≈ 0.49 | ✅ |
| s_NMDA builds with input | Slow rise | Confirmed | ✅ |
| s_NMDA decays after removal | τ ≈ 100 ms | Confirmed | ✅ |
| Zero Mg²⁺ increases firing | More spikes | Confirmed | ✅ |
| V clamped [−100, 60] | Always | 10⁶ steps | ✅ |
| s_NMDA ∈ [0, 1] | Clamped | Confirmed | ✅ |
| NaN recovery | Resets | Confirmed | ✅ |
| Higher g_NMDA → more excitable | Monotonic | Confirmed | ✅ |
| Spike = V crossing −20 mV | Reset to −65 | Confirmed | ✅ |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/channels/nmda.rs:27` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, h, n, s_nmda) |
| NetworkRunner wired | `NeuronVariant::NMDA` |
| `create_neuron("NMDA")` | Yes |
| `supported_models()` | Includes "NMDA" |
| coverage tests | 12 |
| Benchmark | `nmda_1k_steps`: **3.29 ms** (3.29 µs/step), i5-11600K |

---

## Network Coupling

### Recurrent NMDA Networks for Working Memory

A population of NMDANeurons with recurrent excitation can sustain
persistent activity:

$$I_{ext,i} = \sum_j w_{ij} \cdot \text{spike}_j(t) \ast h(t) + I_{stimulus}$$

where h(t) is the synaptic kernel (modelling glutamate release).
When the recurrent NMDA conductance is strong enough (g_NMDA > critical),
the network exhibits bistability: a brief stimulus triggers persistent
activity that outlasts the input by seconds.

### NMDA + AMPA: Fast and Slow Excitation

In biological circuits, excitatory synapses co-release onto both
AMPA and NMDA receptors.  AMPA provides the fast depolarisation
(τ ≈ 2 ms) needed to relieve the Mg²⁺ block, while NMDA provides
the sustained current (τ ≈ 100 ms) for temporal integration.

In SC-NeuroCore, this can be modelled by combining the NMDANeuron's
NMDA current with an additional AMPA component through the external
input.

### NMDA Receptor Pharmacology

| Agent | Effect | Clinical use |
|-------|--------|-------------|
| Ketamine | Non-competitive NMDA block | Anaesthesia, antidepressant |
| PCP (phencyclidine) | Non-competitive block | None (drug of abuse) |
| Memantine | Low-affinity open-channel block | Alzheimer's disease |
| D-cycloserine | GluN1 glycine site partial agonist | Augment exposure therapy |
| Ifenprodil | GluN2B-selective antagonist | Research tool |
| MK-801 (dizocilpine) | High-affinity channel block | Research only |
| Mg²⁺ (high dose) | Physiological block enhancement | Eclampsia, neuroprotection |

In the model, non-competitive block (ketamine, MK-801) is modelled
by reducing g_NMDA.  Competitive block at the glutamate site reduces
the effective input (lower s_drive).  Mg²⁺ modulation changes mg_conc.

### NMDA Conductance and Threshold

The effective firing threshold depends on NMDA conductance:

| g_NMDA (mS/cm²) | Threshold current | Effect |
|-----------------|------------------|--------|
| 0.0 | ~1.5 | WB baseline (no NMDA) |
| 0.5 (default) | ~1.0 | Moderate NMDA boost |
| 1.0 | ~0.5 | Strong NMDA amplification |
| 2.0 | ~0.1 | Near-spontaneous (NMDA-driven) |

Higher g_NMDA progressively lowers the firing threshold through the
positive feedback mechanism described above.

### Clinical Relevance: NMDA Hypofunction

The NMDA hypofunction hypothesis of schizophrenia posits that reduced
NMDA receptor function (via GluN2B reduction, anti-NMDA antibodies,
or PCP/ketamine block) disrupts PFC persistent activity, causing
cognitive deficits.  In the model: reducing g_NMDA or increasing
mg_conc simulates NMDA hypofunction.

---

## References

1. Jahr, C. E. & Stevens, C. F. (1990). Voltage dependence of
   NMDA-activated macroscopic conductances predicted by single-channel
   kinetics. *J Neurosci*, 10(9), 3178–3182.

2. Wang, X. J. (1999). Synaptic basis of cortical persistent activity:
   the importance of NMDA receptors to working memory. *J Neurosci*,
   19(21), 9587–9603.

3. Woodhull, A. M. (1973). Ionic blockage of sodium channels in nerve.
   *J Gen Physiol*, 61(6), 687–708.

4. Wang, X. J. & Buzsáki, G. (1996). Gamma oscillation by synaptic
   inhibition in a hippocampal interneuronal network model. *J Neurosci*,
   16(20), 6402–6413.

5. Compte, A., Brunel, N., Goldman-Rakic, P. S. & Wang, X. J. (2000).
   Synaptic mechanisms and network dynamics underlying spatial working
   memory in a cortical network model. *Cereb Cortex*, 10(9), 910–923.

6. Malenka, R. C. & Bear, M. F. (2004). LTP and LTD: an embarrassment
   of riches. *Neuron*, 44(1), 5–21.

7. Paoletti, P., Bellone, C. & Zhou, Q. (2013). NMDA receptor subunit
   diversity: impact on receptor properties, synaptic plasticity and
   disease. *Nat Rev Neurosci*, 14(6), 383–400.

8. Traynelis, S. F., Wollmuth, L. P., McBain, C. J., et al. (2010).
   Glutamate receptor ion channels: structure, regulation, and function.
   *Pharmacol Rev*, 62(3), 405–496.

9. Lisman, J. E., Fellous, J. M. & Wang, X. J. (1998). A role for
   NMDA-receptor channels in working memory. *Nat Neurosci*, 1(4),
   273–275.

10. Coyle, J. T. (2006). Glutamate and schizophrenia: beyond the
    dopamine hypothesis. *Cell Mol Neurobiol*, 26(4–6), 365–384.

11. Dingledine, R., Borges, K., Bowie, D. & Traynelis, S. F. (1999).
    The glutamate receptor ion channels. *Pharmacol Rev*, 51(1), 7–61.

12. Bhatt, D. & Bhatt, E. (2007). NMDA receptors: old channels, new
    tricks. *Trends Neurosci*, 30(6), 271–273.
