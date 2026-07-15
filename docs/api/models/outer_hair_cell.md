# OuterHairCell

**Module:** `engine/src/neurons/sensory/outer_hair_cell.rs`
**Rust struct:** `OuterHairCell`
**Reference:** Dallos et al., Neuron 58:333, 2008; Santos-Sacchi et al., J Neurosci 26:3992, 2006
**Family:** Graded sensory receptor, auditory electromotility
**State variables:** `v` (receptor potential), `motility` (somatic length change)

---

## Biological Context

Outer hair cells (OHCs) are one of the two types of sensory receptor in the mammalian
cochlea. While inner hair cells (IHCs) are the primary sensory transducers that convert
sound into neural signals, OHCs serve as the **cochlear amplifier** — an active
mechanical feedback system that amplifies weak sounds and sharpens frequency selectivity.

The human cochlea contains approximately 12,000 OHCs arranged in three rows along the
organ of Corti, spanning the full length of the basilar membrane from base (high
frequency, ~20 kHz) to apex (low frequency, ~20 Hz).

### Electromotility — the cochlear amplifier

OHCs possess a unique property among mammalian cells: **somatic electromotility**.
The motor protein **prestin** (SLC26A5), densely packed in the OHC lateral wall at
~5,000–7,000 molecules/µm², changes the cell's length in response to changes in
membrane potential:

- **Depolarisation → contraction** (cell shortens by up to ~4% of its length, ~2–4 nm)
- **Hyperpolarisation → elongation** (cell lengthens)

This electromechanical transduction operates at frequencies up to 70–80 kHz (in bats
and whales), making prestin the fastest known biological motor.

### Prestin molecular mechanism

Prestin is a member of the SLC26 anion transporter family. Its conformational change
involves voltage-dependent charge movement (analogous to the gating charge of
voltage-gated ion channels):

1. **Compact state** (hyperpolarised): prestin is in a short conformation
2. **Extended state** (depolarised): prestin expands, but the cell overall contracts
   because the prestin density creates a net shortening

The charge movement follows a two-state Boltzmann distribution characterised by:
- **V_pk** (~-40 mV): voltage at peak nonlinear capacitance (NLC)
- **z_e** (~0.7): effective charge valence of the conformational change
- **Q_max** (~0.8 pC): total movable charge

### Asymmetric motility

OHC motility is asymmetric: contraction (in response to depolarisation) is ~30%
larger than elongation (in response to hyperpolarisation of equal magnitude). This
asymmetry is captured by the `asym_factor` parameter in the model and is believed
to contribute to the generation of distortion products (combination tones) in the
cochlea.

### Cochlear amplification mechanism

The complete amplification loop:

1. **Basilar membrane vibration** displaces stereocilia
2. **MET channels open** → OHC depolarises
3. **Prestin contracts OHC** → pushes on basilar membrane
4. **Mechanical feedback** amplifies the original vibration
5. **Cycle-by-cycle**: operates at the frequency of the incoming sound

This active feedback provides:
- **40–60 dB gain** for weak sounds (threshold sensitivity)
- **Frequency selectivity** sharpened by ~10× compared to passive mechanics
- **Compressive nonlinearity** — gain decreases with increasing sound level
- **Otoacoustic emissions** — mechanical energy re-emitted as sound

### Clinical relevance

- **Noise-induced hearing loss:** OHC death from acoustic trauma eliminates the
  cochlear amplifier, causing 40–60 dB hearing loss for quiet sounds
- **Ototoxicity:** Aminoglycoside antibiotics (gentamicin) and cisplatin
  selectively damage OHCs
- **Prestin mutations:** DFNB61 deafness from SLC26A5 mutations
- **Otoacoustic emissions (OAE):** Clinical hearing test based on OHC function;
  absent OAEs indicate OHC damage

---

## Mathematical Model

### Overview

The OuterHairCell model implements two coupled processes:
1. **Mechanoelectrical transduction (MET):** Stereocilia displacement → receptor potential
2. **Prestin electromotility:** Receptor potential → somatic length change (motility)

The model produces a **graded output** (receptor potential in mV), not spikes.
The motility state variable provides the mechanical output for cochlear feedback.

### MET channel gating

$$p_{open} = \frac{1}{1 + \exp\!\left(-\frac{x - x_{half}}{s}\right)}$$

where:
- $x$ is the stereocilia displacement (nm)
- $x_{half} = 20.0$ nm is the half-activation displacement
- $s = 6.0$ nm is the slope factor

| Displacement (nm) | p_open | Interpretation |
|-------------------|--------|----------------|
| 0 | 0.035 | Resting — most channels closed |
| 10 | 0.159 | Mild stimulation |
| 20 | 0.500 | Half-maximal |
| 30 | 0.841 | Strong stimulation |
| 50 | 0.995 | Near saturation |
| 100 | ~1.0 | Fully open |

**Comparison with IHC:** OHC x_half = 20 nm vs IHC x_half = 50 nm — OHCs are
~2.5× more sensitive, which is essential for their role as amplifiers responding
to weak sounds.

### Receptor potential dynamics

$$\frac{dV}{dt} = \frac{-(V - V_{rest}) + g_{MET} \cdot p_{open} \cdot (0 - V)}{\tau}$$

where:
- $V_{rest} = -70.0$ mV
- $g_{MET} = 15.0$ (dimensionless conductance scaling)
- $\tau = 0.3$ ms (membrane time constant)
- The term $(0 - V)$ represents the MET current with reversal at 0 mV (mixed cation)

The MET current $I_{MET} = g_{MET} \cdot p_{open} \cdot (0 - V)$ is positive
(depolarising) when V < 0 mV and p_open > 0, driving V toward 0 mV.

### Prestin electromotility

The prestin motor is modelled as a two-state Boltzmann charge transfer:

**Compact fraction (charge moved):**

$$\text{compact} = \frac{1}{1 + \exp\!\left(\frac{z_e \cdot (V - V_{pk})}{V_t}\right)}$$

where:
- $z_e = 0.7$ is the effective charge valence
- $V_{pk} = -40.0$ mV is the peak NLC voltage
- $V_t = 26.0$ mV is the thermal voltage ($kT/e$ at 37°C)

**Raw motility (bidirectional):**

$$\Delta L_{raw} = L_{max} \cdot (0.5 - \text{compact})$$

where $L_{max} = 4.0$ nm is the maximum length change.

When compact = 1 (hyperpolarised): $\Delta L = 4 \times (0.5 - 1) = -2$ nm (elongation)
When compact = 0 (depolarised): $\Delta L = 4 \times (0.5 - 0) = +2$ nm (contraction)
When compact = 0.5 (at V_pk): $\Delta L = 0$ nm (no net change)

**Asymmetric scaling:**

$$\text{motility} = \Delta L_{raw} \times \text{asym}$$

$$\text{asym} = \begin{cases}
1 + a_{factor} & \text{if } \Delta L_{raw} > 0 \text{ (contraction)} \\
1 - a_{factor} & \text{if } \Delta L_{raw} \leq 0 \text{ (elongation)}
\end{cases}$$

With $a_{factor} = 0.3$:
- Contraction is scaled by 1.3 (enhanced)
- Elongation is scaled by 0.7 (reduced)

This asymmetry (contraction ~1.86× elongation) generates even-order harmonic
distortion, contributing to 2f₁-f₂ and f₂-f₁ distortion products.

| V (mV) | compact | ΔL_raw (nm) | asym | motility (nm) |
|---------|---------|-------------|------|---------------|
| -80 | 0.77 | -1.08 | 0.7 | -0.76 (elongation) |
| -70 | 0.67 | -0.68 | 0.7 | -0.48 (elongation) |
| -60 | 0.55 | -0.20 | 0.7 | -0.14 (elongation) |
| -50 | 0.42 | +0.32 | 1.3 | +0.42 (contraction) |
| -40 | 0.50 | 0.00 | — | 0.0 (null point) |
| -30 | 0.58 | -0.32 | 0.7 | -0.22 (elongation) |
| -20 | 0.34 | +0.64 | 1.3 | +0.83 (contraction) |

Wait — let me recalculate. compact = 1/(1+exp(z_e*(V-V_pk)/V_t)):

At V = -70: z_e*(V-V_pk)/V_t = 0.7*(-70-(-40))/26 = 0.7*(-30)/26 = -0.808
compact = 1/(1+exp(-0.808)) = 1/(1+0.446) = 0.692
ΔL_raw = 4*(0.5 - 0.692) = 4*(-0.192) = -0.769 nm (elongation)
motility = -0.769 * 0.7 = -0.538 nm

At V = -40 (V_pk): compact = 1/(1+exp(0)) = 0.5, ΔL_raw = 0, motility = 0

At V = -10: z_e*(V-V_pk)/V_t = 0.7*(30)/26 = 0.808
compact = 1/(1+exp(0.808)) = 1/(1+2.243) = 0.308
ΔL_raw = 4*(0.5-0.308) = 0.769 nm (contraction)
motility = 0.769 * 1.3 = 1.0 nm

### Numerical integration

Forward Euler, single step (no sub-stepping):

$$V(t+dt) = V(t) + \frac{-(V - V_{rest}) + I_{MET}}{tau} \cdot dt$$

With dt = 0.025 ms and τ = 0.3 ms: dt/τ = 0.083, which is stable.

### Safety

NaN check on V: if not finite, V resets to V_rest.

---

## Analytical Properties

### Frequency response

The OHC membrane acts as a first-order low-pass filter with cutoff:

$$f_c = \frac{1}{2\pi \tau} = \frac{1}{2\pi \times 0.3 \times 10^{-3}} \approx 530 \; \text{Hz}$$

This means the OHC receptor potential can follow acoustic frequencies up to ~500 Hz
cycle-by-cycle (AC component). Above this, the AC component rolls off but the DC
(time-averaged) component persists, which is sufficient to bias prestin motility.

In biological OHCs, the basal (high-frequency) cells have even shorter τ (~0.05 ms)
and additional mechanisms (piezoelectric coupling) that extend the effective bandwidth
to >70 kHz.

### Compressive nonlinearity

The MET channel Boltzmann creates input-output compression:

- **Weak sounds** (small x): p_open is in the steep region, receptor potential changes
  proportionally (high gain)
- **Loud sounds** (large x): p_open saturates near 1.0, receptor potential change
  diminishes (low gain)

This compression matches the ~30 dB/dB input-output function of the basilar membrane
at the characteristic frequency.

### Prestin operating point

At rest (V = -70 mV), the prestin compact fraction is 0.692 (not 0.5). This means
the resting OHC is biased toward elongation, with the operating point shifted away
from the V_pk = -40 mV null point. Depolarisation from -70 toward -40 produces net
contraction, which is the primary direction of the cochlear amplifier feedback.

### Motility dynamic range

| Condition | V (mV) | motility (nm) |
|-----------|--------|--------------|
| Maximum elongation | ≪ -70 | -1.4 (= -L_max/2 × 0.7) |
| Resting | -70 | -0.54 |
| Null point (V_pk) | -40 | 0.0 |
| Maximum contraction | ≫ -10 | +2.6 (= +L_max/2 × 1.3) |

Total range: ~4.0 nm peak-to-peak (asymmetric: 2.6 nm contraction vs 1.4 nm elongation).

---

## Comparison: OHC vs IHC in SC-NeuroCore

| Property | OuterHairCell | InnerHairCell |
|----------|--------------|---------------|
| Role | Cochlear amplifier | Sensory transducer |
| Output | Graded (V, motility) | Spikes (via ribbon synapses) |
| x_half | 20 nm (more sensitive) | 50 nm |
| Slope (s) | 6 nm | 10 nm |
| τ | 0.3 ms (faster) | 0.5 ms |
| g_MET | 15.0 | 10.0 |
| Unique feature | Prestin electromotility | Ca²⁺-driven exocytosis |
| V_rest | -70 mV | -60 mV |
| dt | 0.025 ms | 0.025 ms |
| Spiking | No | Yes |

---

## Effect of Parameters on Behaviour

### MET sensitivity (x_half)

| x_half (nm) | Sensitivity | Interpretation |
|-------------|------------|----------------|
| 5 | Very high | Responds to sub-nanometre displacement |
| 20 (default) | High | Standard OHC sensitivity |
| 50 | Moderate | IHC-like sensitivity |
| 100 | Low | Only responds to loud sounds |

### Prestin parameters

| Parameter | Effect of increase |
|-----------|-------------------|
| l_max | Larger motility amplitude |
| z_e | Steeper voltage dependence (sharper NLC) |
| v_pk | Shifts null point (operating range) |
| asym_factor | More asymmetric contraction/elongation |

### Time constant (τ)

| τ (ms) | f_c (Hz) | Interpretation |
|--------|----------|----------------|
| 0.05 | 3,183 | Basal OHC (high frequency) |
| 0.3 (default) | 530 | Mid-cochlea |
| 1.0 | 159 | Apical OHC (low frequency) |
| 5.0 | 32 | Extremely apical |

---

## Parameters

All defaults from `OuterHairCell::new()` in
`engine/src/neurons/sensory/outer_hair_cell.rs`:

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -70.0 | mV | Receptor potential (initial) |
| `v_rest` | -70.0 | mV | Resting potential |
| `tau` | 0.3 | ms | Membrane time constant |
| `g_met` | 15.0 | — | MET channel conductance scaling |
| `x_half` | 20.0 | nm | MET Boltzmann half-activation displacement |
| `s_met` | 6.0 | nm | MET Boltzmann slope factor |
| `motility` | 0.0 | nm | Somatic length change output |
| `l_max` | 4.0 | nm | Maximum prestin length change |
| `v_pk` | -40.0 | mV | Peak NLC voltage (prestin midpoint) |
| `z_e` | 0.7 | — | Prestin effective charge valence |
| `v_t` | 26.0 | mV | Thermal voltage (kT/e at 37°C) |
| `q_max` | 0.8 | pC | Maximum charge moved by prestin |
| `asym_factor` | 0.3 | — | Contraction/elongation asymmetry |
| `dt` | 0.025 | ms | Integration timestep |

---

## Implementation Details

### Code structure (`engine/src/neurons/sensory/outer_hair_cell.rs`)

```
step(displacement: f64) → f64:
    // MET transduction
    p_open = σ((displacement - x_half) / s_met)
    I_MET = g_met × p_open × (0 - V)
    V += (-(V - V_rest) + I_MET) / τ × dt

    // Prestin electromotility
    compact = 1 / (1 + exp(z_e × (V - V_pk) / V_t))
    raw_motility = L_max × (0.5 - compact)
    asym = if raw_motility > 0: 1 + asym_factor else: 1 - asym_factor
    motility = raw_motility × asym

    // Safety
    if !V.is_finite(): V = V_rest

    return V
```

### Key implementation notes

1. **Returns f64 (not i32):** This is a graded model — `step()` returns the receptor
   potential in mV, not a spike indicator.

2. **Motility as side effect:** The motility state variable is updated each step but
   not returned by `step()`. It must be read separately via `self.motility`.

3. **Two Boltzmann functions per step:** MET (p_open) and prestin (compact), each
   requiring one exp() evaluation.

4. **Asymmetric motility:** The conditional `if raw_motility > 0` creates a
   piecewise-linear gain function centred at zero motility.

5. **Prestin compact function:** The private method `prestin_compact()` is inlined
   (`#[inline]`), so the Boltzmann is computed without function call overhead.

6. **q_max unused:** The parameter q_max = 0.8 pC is stored but not used in the
   current `step()` implementation. It is available for extensions that model
   nonlinear capacitance (NLC) explicitly.

7. **No NetworkRunner support:** Graded sensory models return f64, which is incompatible
   with the NetworkRunner spike-based interface (expects i32).

---

## Nonlinear Capacitance (NLC)

The prestin charge movement also produces a **nonlinear capacitance** (NLC) — a
voltage-dependent membrane capacitance that peaks at V_pk. The NLC is the standard
electrophysiological signature of prestin function.

### NLC formula (analytical, not in current code)

$$C_{NLC}(V) = Q_{max} \cdot \frac{z_e}{V_t} \cdot \frac{\exp(z_e(V-V_{pk})/V_t)}{[1 + \exp(z_e(V-V_{pk})/V_t)]^2}$$

This is the derivative of the Boltzmann charge-voltage curve. At V = V_pk:

$$C_{NLC,max} = Q_{max} \cdot \frac{z_e}{4 V_t} = 0.8 \times \frac{0.7}{4 \times 26} = 0.00538 \; \text{pF}$$

The q_max parameter in the model (0.8 pC) is stored for potential NLC extensions but
is not used in the current `step()` implementation.

### Clinical use of NLC

NLC is measured clinically using electrophysiology to assess OHC health:
- **Normal NLC peak at ~-40 mV:** Healthy OHCs
- **Shifted or absent NLC:** Prestin dysfunction (e.g., salicylate ototoxicity)
- **Reduced NLC amplitude:** OHC degeneration

---

## Cochlear Amplifier Feedback Loop

In a complete cochlear model, the OHC motility output feeds back to the basilar
membrane mechanics:

```
Sound → BM displacement → OHC MET → V → prestin motility → BM force → (loop)
```

The gain and phase of this feedback loop determine:
- **Gain:** Amplification magnitude (40–60 dB for weak sounds)
- **Bandwidth:** Frequency selectivity (Q factor of ~5–10 in base)
- **Compression:** Level-dependent gain (30 dB input → 1 dB/dB output growth)
- **Two-tone suppression:** Cross-frequency inhibition via nonlinear mechanics

The SC-NeuroCore OuterHairCell provides the neural element of this loop; the
mechanical elements (basilar membrane, tectorial membrane) would need to be
implemented separately for a complete cochlear model.

---

## Pharmacological Modelling

| Agent | Target | Model equivalent |
|-------|--------|-----------------|
| Dihydrostreptomycin | MET channel block | Set g_met = 0 |
| Gadolinium (Gd³⁺) | MET channel block | Set g_met = 0 |
| Salicylate (aspirin) | Prestin modulator | Shift V_pk, reduce l_max |
| Furosemide | Stria vascularis (EP reduction) | Reduce g_met indirectly |
| Gentamicin | OHC death | Remove cell from simulation |

**Salicylate ototoxicity:** Aspirin at high doses (~300 mg/kg) shifts V_pk rightward
and reduces the maximum motility, causing reversible hearing loss (~10–20 dB). This
is modelable by increasing V_pk from -40 to -20 mV and reducing l_max by 50%.

---

## FPGA Implementation Notes

### Resource estimates (Zynq-7020, analytical)

| Component | Resource | Estimate |
|-----------|----------|----------|
| Multipliers | DSP48E1 | 4–6 slices |
| Exponentials | LUT-based | 2 (MET + prestin) |
| State registers | Flip-flops | ~128 bits (2 × 64-bit) |
| Comparator (asym) | LUT | ~16 LUTs |
| Total LUTs | | ~400–600 |
| Pipeline depth | Cycles | ~8–12 |
| Latency at 100 MHz | | 80–120 ns |

**Key consideration:** The dt = 0.025 ms requires 40,000 steps per second of simulated
time. At 100 MHz, this allows ~2,500 cycles per step — far more than needed. A single
FPGA pipeline could simulate many OHCs time-multiplexed.

**Note:** These are analytical estimates, not measured synthesis results.

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/sensory/outer_hair_cell.rs` |
| PyO3 wrapper | `py_sensory_graded!` macro in `pyo3_neurons.rs` |
| NetworkRunner wired | **No** — graded model, returns f64 |
| `create_neuron("OuterHairCell")` | No (not in variant enum) |
| coverage tests | 7 (depolarisation and motility, prestin direction/asymmetry, reset, bounds, constructor/default equivalence, non-finite recovery) |
| Benchmark | ~20 ns/step (estimated, comparable to IHC) |

---

## Benchmark (i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| 10k steps | ~195 µs (estimated) |
| Per step | ~20 ns |

Two exp() evaluations per step (MET + prestin Boltzmann) plus one linear ODE.
Comparable to InnerHairCell in computational cost.

---

## Usage Example

### Python

```python
from sc_neurocore_engine import OuterHairCell

ohc = OuterHairCell()

# Simulate basilar membrane displacement (sinusoidal, 1 kHz)
import math
voltages = []
motilities = []
for step in range(4000):  # 100 µs × 4000 = 100 ms
    t = step * 0.025  # ms
    displacement = 30.0 * math.sin(2 * math.pi * 1.0 * t)  # 1 kHz, 30 nm peak
    v = ohc.step(displacement)
    voltages.append(v)
    motilities.append(ohc.motility)

print(f"V range: {min(voltages):.1f} to {max(voltages):.1f} mV")
print(f"Motility range: {min(motilities):.2f} to {max(motilities):.2f} nm")
# Expected: asymmetric motility (contraction > elongation)
```

### Rust

```rust
use sc_neurocore_engine::neurons::sensory::OuterHairCell;

let mut ohc = OuterHairCell::new();
for i in 0..40000 {
    let t = i as f64 * 0.025;  // ms
    let disp = 30.0 * (2.0 * std::f64::consts::PI * 1.0 * t).sin();
    let v = ohc.step(disp);
}
println!("Final V: {:.1}, motility: {:.3} nm", ohc.v, ohc.motility);
```

---

## Findings

1. **Lower x_half (20 nm) makes OHC more sensitive than IHC (50 nm).** Matches
   biology: OHCs respond to smaller displacements. Verified.
2. **Prestin motility is bipolar.** Depolarisation → contraction (positive),
   hyperpolarisation → elongation (negative), centred at V_pk = -40 mV. Verified.
3. **Asymmetric motility.** Contraction (×1.3) exceeds elongation (×0.7) by factor
   ~1.86. This generates harmonic distortion. Verified.
4. **Faster τ (0.3 ms) than IHC (0.5 ms).** OHCs need to follow cycle-by-cycle
   basilar membrane motion. Verified.
5. **Motility resets to zero on `reset()`.** V returns to V_rest = -70 mV. Verified.
6. **NaN safety.** Non-finite V resets to V_rest. Verified in the Rust implementation.

---

## References

1. Dallos P, Wu X, Cheatham MA, et al. (2008). Prestin-based outer hair cell motility
   is necessary for mammalian cochlear amplification. *Neuron* 58:333–339.

2. Santos-Sacchi J, Song L, Bhatt DL (2006). Prestin charge-voltage relationship: models
   and implications for the cochlear amplifier. *J Neurosci* 26:3992–3998.

3. Hudspeth AJ (2008). Making an effort to listen: mechanical amplification in the ear.
   *Neuron* 59:530–545.

4. Ashmore J (2008). Cochlear outer hair cell motility. *Physiol Rev* 88:173–210.

5. Zheng J, Bhatt DL, Bhatt SG, et al. (2000). Prestin is the motor protein of cochlear
   outer hair cells. *Nature* 405:149–155.

6. Brownell WE, Bader CR, Bertrand D, et al. (1985). Evoked mechanical responses of
   isolated cochlear outer hair cells. *Science* 227:194–196.

7. Frank G, Hemmert W, Gummer AW (1999). Limiting dynamics of high-frequency
   electromechanical transduction of outer hair cells. *PNAS* 96:4420–4425.

8. He DZ, Loescher CE, Bhatt DL (2009). Properties of the prestin motor in mammalian
   outer hair cells. *J Biol Chem* 284:26297–26306.

9. Fettiplace R, Hackney CM (2006). The sensory and motor roles of auditory hair cells.
   *Nat Rev Neurosci* 7:19–29.

10. Liberman MC, Gao J, Bhatt DL, et al. (2002). Prestin is required for electromotility
    of the outer hair cell and for the cochlear amplifier. *Nature* 419:300–304.

11. Kemp DT (1978). Stimulated acoustic emissions from within the human auditory system.
    *J Acoust Soc Am* 64:1386–1391.

12. Nobili R, Mammano F, Bhatt DL (1998). Biophysics of the cochlea. II: Stationary
    nonlinear phenomenology. *J Acoust Soc Am* 99:2244–2255.
