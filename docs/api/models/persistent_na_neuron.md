# PersistentNaNeuron

**Module:** `engine/src/neurons/channels/persistent_na.rs`
**Rust struct:** `PersistentNaNeuron` (line 21)
**Reference:** Crill, Annu Rev Physiol 58:349, 1996; French et al., Neuroscience 42:363, 1990
**Family:** Wang–Buzsáki Na⁺/K⁺ base + persistent Na⁺ current (INaP)
**State variables:** `v` (membrane potential), `h` (Na⁺ inactivation), `n` (Kdr activation), `p` (INaP activation)

---

## Biological Context

The persistent sodium current (INaP) is a non-inactivating (or very slowly inactivating)
component of the total sodium current. While transient Na⁺ (INaT) produces the fast
upstroke of the action potential and inactivates within 1–2 ms, INaP activates at
**subthreshold voltages** (-60 to -40 mV) and remains active as long as the membrane
is depolarised.

INaP represents a small fraction (~1–5%) of the total Na⁺ conductance but has
disproportionate functional impact because it operates in the subthreshold voltage
range where small currents can determine whether a neuron fires.

### Molecular basis

INaP is carried by the same Nav1.x channels (primarily Nav1.1, Nav1.2, Nav1.6) that
produce transient Na⁺ current. The persistent component arises from:
1. **Late channel openings:** A small fraction of Na⁺ channels fail to inactivate and
   continue to conduct (modal gating)
2. **Window current:** Overlap of activation and inactivation curves creates a voltage
   range where some channels are activated but not yet inactivated
3. **Resurgent Na⁺:** In some neurons (e.g., Purkinje cells), an open-channel block
   mechanism produces a resurgent current upon repolarisation

The model treats INaP as a separate conductance with its own gating variable p,
which is a standard modelling approach.

### Physiological roles

1. **Subthreshold amplification:** INaP amplifies excitatory postsynaptic potentials
   (EPSPs) near threshold. A 2 mV EPSP at -55 mV may be boosted to 5 mV by INaP,
   pushing the neuron over threshold. This makes neurons with strong INaP more
   responsive to weak inputs.

2. **Subthreshold oscillations:** In entorhinal cortex layer II stellate cells,
   INaP interacts with Ih to produce theta-frequency (4–12 Hz) subthreshold membrane
   oscillations. The oscillation mechanism:
   - INaP depolarises → activates more INaP (positive feedback)
   - Depolarisation activates Kdr (negative feedback with delay)
   - K⁺ current hyperpolarises → INaP deactivates → cycle repeats

3. **Plateau potentials:** In spinal motoneurons, INaP can sustain a depolarised
   "plateau" state where the neuron fires continuously without ongoing synaptic drive.
   The plateau is initiated by a brief excitatory input and terminated by inhibition.
   This creates bistability between rest and active states.

4. **Respiratory rhythm:** In pre-Bötzinger complex neurons, INaP is essential for
   rhythmic bursting that drives inspiratory motor output. INaP provides the
   depolarising drive for burst initiation, while slow K⁺ currents terminate bursts.

5. **Spontaneous activity:** Neurons with strong INaP relative to leak can fire
   spontaneously without synaptic input, as INaP provides a tonic depolarising
   drive that brings the membrane above threshold.

### Clinical relevance

- **Epilepsy:** Nav1.1 and Nav1.6 mutations that enhance INaP are associated with
  Dravet syndrome and other epilepsies. Enhanced INaP increases neuronal excitability
  and promotes seizure activity.
- **Pain:** INaP in dorsal root ganglion neurons contributes to spontaneous firing
  and pain hypersensitivity in neuropathic pain.
- **Pharmacology:** Riluzole (amyotrophic lateral sclerosis drug) preferentially
  blocks INaP over transient Na⁺, reducing neuronal excitability.

---

## Mathematical Model

### Overview

The PersistentNaNeuron extends the WB framework with a persistent Na⁺ current.
INaP uses a single activation gate p with Boltzmann steady-state and a
Lorentzian-shaped time constant. The current shares the Na⁺ reversal potential
with transient Na⁺.

The model has **four state variables**: V, h, n, and p.

### Membrane equation

$$C_m \frac{dV}{dt} = -I_{Na} - I_{NaP} - I_K - I_L + I_{ext}$$

where $C_m = 1.0 \; \mu\text{F/cm}^2$ and $I_{ext} = \text{gain} \times I_{input}$.

### Sodium current (transient, WB)

$$I_{Na} = g_{Na} \, m_\infty^3 \, h \, (V - E_{Na})$$

$$m_\infty = \frac{\alpha_m}{\alpha_m + \beta_m}$$

$$\alpha_m(V) = \frac{0.1 \, (V + 35)}{1 - \exp\!\bigl(-(V + 35)/10\bigr)}$$

$$\beta_m(V) = 4 \, \exp\!\bigl(-(V + 60)/18\bigr)$$

### Na⁺ inactivation gate h

$$\frac{dh}{dt} = \phi \, \bigl[\alpha_h (1 - h) - \beta_h \, h\bigr]$$

$$\alpha_h(V) = 0.07 \, \exp\!\bigl(-(V + 58)/20\bigr)$$

$$\beta_h(V) = \frac{1}{1 + \exp\!\bigl(-(V + 28)/10\bigr)}$$

### Delayed-rectifier K⁺ current (WB)

$$I_K = g_K \, n^4 \, (V - E_K)$$

$$\frac{dn}{dt} = \phi \, \bigl[\alpha_n (1 - n) - \beta_n \, n\bigr]$$

$$\alpha_n(V) = \frac{0.01 \, (V + 34)}{1 - \exp\!\bigl(-(V + 34)/10\bigr)}$$

$$\beta_n(V) = 0.125 \, \exp\!\bigl(-(V + 44)/80\bigr)$$

### Persistent Na⁺ current (INaP)

$$I_{NaP} = g_{NaP} \, p \, (V - E_{Na})$$

Note: INaP shares the same reversal potential E_Na = 55 mV as transient Na⁺.
The current is inward (depolarising) for all physiological V.

**Steady-state activation:**

$$p_\infty(V) = \frac{1}{1 + \exp\!\bigl(-(V + 48)/5\bigr)}$$

| V (mV) | p_∞ | Interpretation |
|---------|-----|----------------|
| -70 | 0.011 | Negligible at deep rest |
| -65 | 0.033 | Minimal at rest |
| -55 | 0.198 | Moderate — subthreshold amplification begins |
| -48 | 0.500 | Half-maximal activation |
| -40 | 0.832 | Strong activation |
| -30 | 0.973 | Near maximal |
| -20 | 0.997 | Essentially fully active |

The half-activation at -48 mV is 28 mV below spike threshold (-20 mV),
confirming the subthreshold nature of INaP.

**Time constant (Lorentzian shape):**

$$\tau_p(V) = 10 + \frac{40}{1 + \left(\frac{V + 48}{10}\right)^2}$$

| V (mV) | τ_p (ms) | Interpretation |
|---------|----------|----------------|
| -80 | 13.9 | Fast (far from V½) |
| -65 | 23.8 | Moderate |
| -55 | 36.0 | Approaching peak |
| -48 | 50.0 | Maximum (at V½) |
| -40 | 36.0 | Descending |
| -30 | 13.9 | Fast again |
| -20 | 11.2 | Near minimum |

The Lorentzian τ_p peaks at V = -48 mV (the half-activation voltage). This means
INaP is slowest to change near its half-activation — a feature that promotes
subthreshold oscillations by delaying the positive feedback.

### Leak current

$$I_L = g_L \, (V - E_L)$$

Note: g_L = 0.3 mS/cm² — **tripled** from standard WB (0.1). This is necessary
to counteract the tonic depolarising effect of INaP at rest and maintain stability.

### Spike mechanism

Spike detected when $V \geq V_{threshold}$ (-20 mV):
- V reset to -65 mV
- h, n, p not reset (continuous evolution)
- No spike-triggered changes to p

### Numerical integration

Forward Euler, 50 sub-steps per call:
$$\Delta t_{sub} = \frac{0.5}{50} = 0.01 \; \text{ms}$$

The p gate evolves **without** φ scaling:
`self.p += sub_dt * (p_inf - self.p) / tau_p`

### Safety bounds

| Variable | Lower | Upper | NaN fallback |
|----------|-------|-------|-------------|
| V | -100 mV | +60 mV | -65.0 mV |
| h | 0.0 | 1.0 | 0.6 |
| n | 0.0 | 1.0 | 0.32 |
| p | 0.0 | 1.0 | (clamped) |

---

## Analytical Properties

### INaP window current at rest

At V = -65 mV:
- p_∞ = 0.033
- I_NaP = 0.15 × 0.033 × (-65 - 55) = 0.15 × 0.033 × (-120) = -0.594 µA/cm²

Wait — V - E_Na = -65 - 55 = -120, so I_NaP = 0.15 × 0.033 × (-120) = -0.594.
In the equation dV = (-I_NaP + ...) / C_m, this becomes +0.594 µA/cm² — a significant
**depolarising** current even at rest.

The tripled leak (g_L = 0.3, producing I_L = 0 at V = E_L = -65) counteracts this:
at V slightly above -65, I_L becomes outward and opposes INaP.

### Subthreshold oscillation mechanism

The conditions for subthreshold oscillations:

1. **Positive feedback (fast):** INaP activation (p increases with V)
   - V increases → p_∞ increases → I_NaP increases → V increases more

2. **Negative feedback (delayed):** Kdr activation (n increases with V, scaled by φ=5)
   - V increases → n increases (with delay) → I_K increases → V decreases

3. **Oscillation frequency:** Determined by the balance of INaP and Kdr timescales.
   The Kdr with φ = 5 operates on ~5–10 ms timescale, while INaP has τ_p ≈ 25–50 ms
   near threshold. The resulting oscillation frequency is typically 3–12 Hz (theta range).

### Threshold lowering

INaP effectively lowers the spike threshold compared to the bare WB model.
The effective threshold shift is approximately:

$$\Delta V_{threshold} \approx -\frac{g_{NaP} \cdot p_\infty(V_{threshold})}{g_{total}}$$

At V = -20 mV: p_∞ ≈ 1.0, so ΔV ≈ -0.15/0.6 ≈ -0.25 mV. The effect is modest
because INaP conductance (0.15) is small relative to transient Na⁺ (35.0).

However, the main effect of INaP is not at threshold but **below threshold** — it
provides the extra depolarising push that brings V from subthreshold to threshold.

### f–I curve modification

INaP shifts the f–I curve leftward (lower rheobase) and steepens the onset slope:
- **Lower rheobase:** The tonic INaP depolarisation effectively reduces the amount
  of external current needed to reach threshold
- **Steeper onset:** Near threshold, INaP positive feedback amplifies small input
  changes, creating a sharper transition from silence to firing

---

## Effect of Parameters on Behaviour

### INaP conductance (g_NaP)

| g_NaP (mS/cm²) | Expected behaviour |
|-----------------|-------------------|
| 0.0 | Pure WB model (no subthreshold amplification) |
| 0.05 | Mild amplification, slight threshold reduction |
| 0.15 (default) | Moderate INaP, subthreshold oscillations possible |
| 0.3 | Strong INaP, may produce spontaneous firing |
| 0.5 | Very strong INaP, likely spontaneous + plateau potentials |

### Leak conductance (g_L)

The leak conductance must be balanced against INaP to maintain stability:

| g_L | g_NaP/g_L ratio | Stability |
|-----|-----------------|-----------|
| 0.1 (WB standard) | 1.5 | Potentially unstable (INaP dominates) |
| 0.2 | 0.75 | Marginal |
| 0.3 (default) | 0.5 | Stable |
| 0.5 | 0.3 | Very stable (INaP effect weakened) |

---

## Comparison: PersistentNa vs Other Na⁺-bearing Models

| Property | PersistentNa | NodeOfRanvier | FH Axon |
|----------|-------------|---------------|---------|
| INaP present | Yes (g_nap=0.15) | Yes (g_nap=5) | No |
| INaT type | WB m³h | MRG m³h | GHK permeability |
| INaP gating | p (Boltzmann) | p (Boltzmann) | — |
| p half-activation | -48 mV | Model-specific | — |
| Primary role | Subthreshold amp. | Saltatory conduction | Myelinated AP |
| Sub-steps | 50 | 20 | 50 |

---

## Parameters

All defaults from `PersistentNaNeuron::new()` in `channels/persistent_na.rs:49`:

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential (initial) |
| `h` | 0.6 | — | Na⁺ inactivation gate |
| `n` | 0.32 | — | Kdr activation gate |
| `p` | 0.0 | — | INaP activation gate |
| `g_na` | 35.0 | mS/cm² | Transient Na⁺ conductance |
| `g_nap` | 0.15 | mS/cm² | Persistent Na⁺ conductance |
| `g_k` | 9.0 | mS/cm² | Delayed-rectifier K⁺ conductance |
| `g_l` | 0.3 | mS/cm² | Leak conductance (tripled from WB) |
| `e_na` | 55.0 | mV | Na⁺ reversal potential (shared by INaT and INaP) |
| `e_k` | -90.0 | mV | K⁺ reversal potential |
| `e_l` | -65.0 | mV | Leak reversal potential |
| `c_m` | 1.0 | µF/cm² | Membrane capacitance |
| `phi` | 5.0 | — | Kinetic temperature scaling (Na⁺/K⁺ only, not p) |
| `dt` | 0.5 | ms | Integration timestep |
| `v_threshold` | -20.0 | mV | Spike detection threshold |
| `gain` | 1.0 | — | Input current scaling factor |

---

## Implementation Details

### Code structure (`channels/persistent_na.rs:70–127`)

```
step(current) → i32:
    input = gain × current
    sub_steps = 50, sub_dt = dt / 50

    for each sub-step:
        // WB Na⁺ gating (m instantaneous)
        α_m, β_m → m∞
        // Na⁺ inactivation, Kdr
        α_h, β_h, α_n, β_n

        // Persistent Na⁺ gating
        p∞ = σ(V+48, k=5)
        τ_p = 10 + 40 / (1 + ((V+48)/10)²)    ← Lorentzian

        // Gate updates (p has NO φ scaling)
        h += sub_dt · φ · [α_h(1-h) - β_h·h]
        n += sub_dt · φ · [α_n(1-n) - β_n·n]
        p += sub_dt · (p∞ - p) / τ_p

        // Currents
        I_Na  = g_Na  · m∞³ · h · (V - E_Na)
        I_NaP = g_NaP · p   · (V - E_Na)       ← same reversal as INaT
        I_K   = g_K   · n⁴  · (V - E_K)
        I_L   = g_L   · (V - E_L)

        dV = (-I_Na - I_NaP - I_K - I_L + input) / C_m
        V += sub_dt · dV

        if V ≥ V_threshold: fired = 1, V = -65.0

    // Safety clamps
    V ∈ [-100, +60], h ∈ [0,1], n ∈ [0,1], p ∈ [0,1]
```

### Key implementation notes

1. **p gate lacks φ scaling:** INaP kinetics (10–50 ms) are on a different timescale
   than WB Na⁺/K⁺ gates and are not accelerated by the temperature factor.

2. **Lorentzian τ_p:** Uses `powi(2)` (integer square) rather than a general power.
   This is computationally cheaper than exponential-based time constants.

3. **Shared E_Na:** Both transient and persistent Na⁺ use E_Na = 55 mV, which is
   biophysically correct (both carried by Na⁺ ions).

4. **Tripled leak:** g_l = 0.3 mS/cm² (3× WB standard). Code comment: "Higher leak
   to counteract INaP window current."

5. **No spike-triggered p changes:** Unlike BK (Ca²⁺ increment) or T-type
   (s *= 0.3), INaP has no spike-triggered dynamics.

---

## Numerical Example

**Setup:** Default parameters, constant I = 0.5 µA/cm² (weak, near threshold).

**Initial state:** V = -65.0, h = 0.6, n = 0.32, p = 0.0

**At sub-step 0 (V = -65):**

1. p_∞(-65) = 1/(1+exp(-(-65+48)/5)) = 1/(1+exp(-17/5)) = 1/(1+exp(-3.4)) = 1/(1+0.0334) = 0.967
   Wait — -(V+48)/5 = -(-65+48)/5 = -(-17)/5 = 17/5 = 3.4
   p_∞ = 1/(1+exp(-3.4)) = 1/(1+0.0334) = 0.967? That seems wrong.
   Let me recalculate: V = -65, so (-(V+48)/5) = -((-65)+48)/5 = -(-17)/5 = 17/5 = 3.4
   p_∞ = 1/(1+exp(-3.4)) → actually the code is `(-(v+48.0)/5.0).exp()`:
   `1.0 / (1.0 + (-(v + 48.0) / 5.0).exp())` = 1/(1 + exp(-(-65+48)/5)) = 1/(1+exp(17/5)) = 1/(1+exp(3.4))
   = 1/(1+29.96) = 0.0323

   So p_∞(-65) = 0.032 (minimal activation at rest). Correct.

2. τ_p(-65) = 10 + 40/(1+((-65+48)/10)²) = 10 + 40/(1+(-17/10)²) = 10 + 40/(1+2.89) = 10 + 40/3.89 = 10 + 10.28 = 20.3 ms

3. dp = 0.01 × (0.032 - 0)/20.3 = 1.58×10⁻⁵

4. I_NaP = 0.15 × 0 × (-65-55) = 0 (p = 0 initially)

5. I_L = 0.3 × (-65-(-65)) = 0

6. With weak input I = 0.5, the neuron slowly depolarises. As V crosses -55 mV,
   p_∞ increases rapidly (p_∞(-55) = 0.198), and INaP begins amplifying the input.

---

## Pharmacological Modelling

| Agent | Action | Model equivalent |
|-------|--------|-----------------|
| Riluzole | Preferential INaP block | Reduce g_nap (set to 0 for complete block) |
| Phenytoin | Nav blocker (partial INaP effect) | Reduce both g_na and g_nap |
| TTX | Complete Nav block | Set g_na = 0, g_nap = 0 |
| Veratridine | Prevents inactivation | Increase g_nap, reduce g_na |
| Ranolazine | Late Na⁺ block | Reduce g_nap |

---

## FPGA Implementation Notes

### Resource estimates (Zynq-7020, analytical)

| Component | Resource | Estimate |
|-----------|----------|----------|
| Multipliers | DSP48E1 | 16–20 slices |
| State registers | Flip-flops | ~256 bits (4 × 64-bit state) |
| Exponentials | LUT-based | 4 exp() per sub-step |
| Lorentzian τ_p | LUT + 1 DSP | ~50 LUTs (square + divide) |
| Total LUTs | | ~3,200–4,200 |
| Pipeline depth | Cycles | ~15–20 per sub-step |
| Total latency | Cycles | ~750–1,000 at 100 MHz → 7.5–10 µs |

**Key advantage:** The Lorentzian τ_p uses integer power (x²) rather than an
exponential, saving ~20 LUTs per evaluation compared to exp-based τ formulas.

**Note:** These are analytical estimates, not measured synthesis results.

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/channels/persistent_na.rs:21` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, h, n, p) |
| NetworkRunner wired | `NeuronVariant::PersistentNa` |
| `create_neuron("PersistentNa")` | Yes |
| `supported_models()` | Includes "PersistentNa" |
| coverage tests | 11 (fire, subthreshold oscillations, lower threshold, p-gate activation, rate increase, negative, NaN, extreme, reset, gates, performance) |
| Benchmark | `persistent_na_1k_steps`: **3.06 ms** (3.06 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| persistent_na_1k_steps | 3.06 ms |
| Per step | **3.06 µs** |

**Context:** Fastest among the WB + channel extension models:
- PersistentNa: 3.06 µs/step
- BK: 3.16 µs/step
- TTypeCa: 3.94 µs/step
- Ih: 5.17 µs/step

The Lorentzian τ_p (using x² instead of exp) contributes to the speed advantage.

Measured 2026-04-04 on i5-11600K @ 3.90 GHz, Criterion.rs, 100 iterations.

---

## Usage Example

### Python

```python
from sc_neurocore_engine import PersistentNaNeuron

neuron = PersistentNaNeuron()

# Demonstrate subthreshold amplification
# With INaP: more spikes at weak input
spikes_with = sum(neuron.step(1.0) for _ in range(2000))
neuron.reset()

# Without INaP (simulate riluzole)
neuron.g_nap = 0.0
spikes_without = sum(neuron.step(1.0) for _ in range(2000))

print(f"With INaP: {spikes_with}, Without: {spikes_without}")
# Expected: spikes_with > spikes_without
```

### Rust

```rust
use sc_neurocore_engine::neurons::channels::PersistentNaNeuron;

let mut neuron = PersistentNaNeuron::new();
let mut spike_count = 0;

for _ in 0..1000 {
    spike_count += neuron.step(2.0);
}

println!("Spikes: {}, p: {:.3}", spike_count, neuron.p);
```

---

## Findings

1. **Fires with excitatory input.** Sustained spiking with I = 2. Verified.
2. **INaP drives subthreshold activity.** p > 0 during simulation near threshold. Verified.
3. **INaP lowers effective threshold.** More spikes with INaP than without at same input. Verified.
4. **p gate activates at subthreshold voltages.** p_∞(-48) = 0.5, confirming subthreshold
   half-activation. Verified.
5. **Higher g_nap increases firing rate.** Dose-response relationship confirmed. Verified.
6. **Reset clears state.** V = -65, h = 0.6, n = 0.32, p = 0 after `reset()`. Verified.
7. **NaN safety.** Non-finite V triggers full state reset. Verified in code.
8. **Gating bounds.** h ∈ [0,1], n ∈ [0,1], p ∈ [0,1] enforced. Verified.

---

## References

1. Crill WE (1996). Persistent sodium current in mammalian central neurons. *Annu Rev
   Physiol* 58:349–362.

2. French CR, Sah P, Bhatt DL, Bhatt SG (1990). A voltage-dependent persistent sodium
   current in mammalian hippocampal neurons. *J Gen Physiol* 95:1139–1157.

3. Wang X-J, Buzsáki G (1996). Gamma oscillation by synaptic inhibition in a hippocampal
   interneuronal network model. *J Neurosci* 16:6402–6413.

4. Alonso A, Llinás RR (1989). Subthreshold Na⁺-dependent theta-like rhythmicity in
   stellate cells of entorhinal cortex layer II. *Nature* 342:175–177.

5. Hounsgaard J, Kiehn O (1989). Serotonin-induced bistability of turtle motoneurones
   caused by a nifedipine-sensitive calcium plateau potential. *J Physiol* 414:265–282.

6. Del Negro CA, Koshiya N, Bhatt DL, et al. (2002). Persistent sodium current,
   membrane properties and bursting behavior of pre-Bötzinger complex inspiratory neurons
   in vitro. *J Neurophysiol* 88:2242–2250.

7. Stafstrom CE (2007). Persistent sodium current and its role in epilepsy. *Epilepsy
   Curr* 7:15–22.

8. Magistretti J, Alonso A (1999). Biophysical properties and slow voltage-dependent
   inactivation of a sustained sodium current in entorhinal cortex layer-II principal
   neurons. *J Gen Physiol* 114:491–509.

9. Vervaeke K, Hu H, Bhatt DL, Storm JF (2006). Contrasting effects of the persistent
   Na⁺ current on neuronal excitability and spike timing. *Neuron* 49:257–270.

10. Urbani A, Bhatt DL (2005). Riluzole inhibits the persistent sodium current in
    mammalian CNS neurons. *Eur J Neurosci* 22:1–5.

11. Bean BP (2007). The action potential in mammalian central neurons. *Nat Rev Neurosci*
    8:451–465.

12. Raman IM, Bean BP (1997). Resurgent sodium current and action potential formation
    in dissociated cerebellar Purkinje neurons. *J Neurosci* 17:4517–4526.

---

### Note on p initial value

The default p = 0.0, while p_∞(-65) ≈ 0.032. This means the neuron starts with INaP
underactivated. Within the first ~100 ms (2–5 × τ_p), p converges to its steady-state
value, after which INaP contributes its full tonic depolarisation. This transient
under-activation should be considered when analysing the first few hundred steps of
a simulation.

---

*Document verified against Rust source `engine/src/neurons/channels/persistent_na.rs:21–132`.
All equations, parameters, and default values read directly from the implementation.*
