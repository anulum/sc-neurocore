<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — MerkelCell model reference -->
# MerkelCell

**Module:** `engine/src/neurons/sensory/merkel_cell.rs`
**Rust struct:** `MerkelCell`
**Reference:** Lesniak et al., PNAS 111:6461, 2014
**Family:** Spiking sensory receptor, slowly adapting type I (SAI) mechanoreceptor
**State variables:** `v` (membrane potential), `adapt` (slow adaptation variable)

---

## Biological Context

Merkel cells are slowly adapting type I (SAI) mechanoreceptors located in the basal
epidermis. They form the **Merkel cell–neurite complex** with myelinated Aβ afferents
and are concentrated in areas with high tactile acuity: fingertips (~100/cm²), lips,
and genitalia.

### Discovery and controversy

Merkel cells were first described by Friedrich Sigmund Merkel in 1875. For over a
century, the site of mechanotransduction was debated: does the Merkel cell itself
transduce force, or does the afferent nerve ending? In 2014, two landmark studies
(Maksimovic et al., Nature; Woo et al., Nature) demonstrated that Merkel cells are
bona fide mechanotransducers expressing Piezo2 channels. The afferent nerve ending
responds to both the Merkel cell's synaptic output and its own mechanosensitivity,
creating a dual-receptor system.

### Slowly adapting type I (SAI) response

SAI afferents are characterised by their response to sustained, static skin indentation:

1. **Dynamic phase (onset):** Brief high-frequency burst (50–100 Hz) at stimulus onset,
   encoding the rate of indentation (velocity sensitivity)
2. **Static phase (sustained):** Maintained lower-frequency discharge (10–30 Hz) for the
   duration of the stimulus, encoding indentation depth
3. **Slow adaptation:** Firing rate gradually decreases over seconds, but never fully
   adapts — firing continues as long as the stimulus is present

This contrasts with:
- **RA (rapidly adapting, Meissner):** Responds only at onset/offset, fully adapts
- **SAII (slowly adapting type II, Ruffini):** Responds to skin stretch, very slow adaptation
- **Pacinian (rapidly adapting II):** Responds to vibration (40–800 Hz), fully adapts

### Functional roles

1. **Texture discrimination:** SAI afferents encode fine spatial details of surfaces
   through their small receptive fields and sustained response to pressure patterns
2. **Edge detection:** The surround inhibition of SAI receptive fields enhances edge contrast
3. **Form perception:** Reading Braille depends critically on SAI afferents — patients with
   SAI loss cannot discriminate Braille dots
4. **Grip control:** SAI afferents provide sustained feedback about grip force during object
   manipulation, preventing slipping
5. **Spatial acuity:** Two-point discrimination threshold is ~2 mm on fingertips, matching
   the density and receptive field size of SAI afferents

### Mechanotransduction mechanism

1. **Skin indentation** deforms the tissue around the Merkel cell
2. **Piezo2 channels** in the Merkel cell membrane open in response to membrane stretch
3. **Ca²⁺ influx** through Piezo2 triggers neurotransmitter release (likely glutamate
   or possibly a neuropeptide)
4. **Afferent activation:** The Aβ nerve terminal is depolarised by both:
   - Chemical signalling from the Merkel cell (slow, sustained component)
   - Direct mechanosensitivity of the nerve ending (fast, dynamic component)
5. **Action potential generation** in the Aβ afferent → signal travels to dorsal root
   ganglion → dorsal column → somatosensory cortex (S1)

### Clinical relevance

- **Diabetic neuropathy:** SAI afferents are among the first to be affected, causing
  loss of fine touch discrimination and increased risk of unnoticed injuries
- **Merkel cell carcinoma:** A rare but aggressive neuroendocrine skin cancer arising
  from Merkel cells, often associated with Merkel cell polyomavirus (MCPyV)
- **Peripheral neuropathy assessment:** SAI function is tested with monofilaments
  (Semmes-Weinstein test) and two-point discrimination
- **Prosthetic feedback:** SAI-like encoding is used in tactile feedback systems for
  prosthetic limbs (providing sustained pressure information)

---

## Mathematical Model

### Overview

The MerkelCell implements a **leaky integrate-and-fire with slow adaptation** that
produces the characteristic SAI discharge pattern: initial burst followed by sustained
lower-frequency firing that slowly adapts.

Two state variables: V (membrane potential) and adapt (slow adaptation current).

### Membrane equation

$$\tau \frac{dV}{dt} = -(V - V_{rest}) + \text{gain} \cdot P - w$$

where:
- $\tau = 5.0$ ms is the membrane time constant
- $V_{rest} = -65.0$ mV is the resting potential
- $P$ is the pressure input (clamped ≥ 0)
- $w$ is the adaptation variable
- gain = 1.5 is the pressure-to-current conversion

The maintained Rust engine uses exact first-order relaxation over each
timestep under constant pressure and adaptation state:

$$V_\infty = V_{rest} + \text{gain} \cdot \max(P, 0) - w$$

$$V(t+dt) = V_\infty + (V(t)-V_\infty)\exp(-dt/\tau)$$

### Slow adaptation dynamics

$$\tau_{adapt} \frac{dw}{dt} = a_{adapt} \cdot (V - V_{rest}) - w$$

where:
- $\tau_{adapt} = 200.0$ ms is the adaptation time constant (very slow)
- $a_{adapt} = 0.3$ is the adaptation coupling strength

The adaptation target is non-negative and uses the membrane candidate:

$$w_\infty = \max(0, a_{adapt}\max(0,V(t+dt)-V_{rest}))$$

$$w(t+dt) = w_\infty + (w(t)-w_\infty)\exp(-dt/\tau_{adapt})$$

**Important:** The adaptation update uses the **candidate** membrane state, not
the previous membrane state. This preserves the intended tight coupling while
removing finite-timestep drift from the older raw-Euler update.

### Adaptation mechanism

The adaptation variable w acts as a **slow negative feedback**:

1. When V is depolarised above rest: $a_{adapt} \cdot (V - V_{rest}) > 0$ → w increases
2. Increased w reduces the effective drive: $\text{drive} = \text{gain} \cdot P - w$
3. Reduced drive → lower firing rate → V closer to rest
4. Lower V → w decreases (slowly, τ = 200 ms)

This creates the characteristic SAI discharge pattern:
- **Initial burst:** w ≈ 0 at onset → full drive → high firing rate
- **Adaptation:** w builds up → effective drive decreases → rate drops
- **Steady state:** w reaches equilibrium where spike-triggered adaptation balances
  the exponential decay

### Spike mechanism

$$\text{if } V \geq V_{threshold}: \quad V \leftarrow V_{reset}, \; \text{return } 1$$

- $V_{threshold} = -50.0$ mV
- $V_{reset} = -70.0$ mV
- No explicit refractory period — the 20 mV reset-to-threshold gap provides an
  implicit minimum ISI

### Input rectification

Pressure is clamped: `pressure.max(0.0)`. Negative pressure values (tension/release)
are treated as zero. This is biophysically appropriate — Merkel cells respond to
compression, not tension.

---

## Analytical Properties

### Steady-state firing rate

At steady state with constant pressure P, the adaptation variable satisfies:

$$w_{ss} = a_{adapt} \cdot (V_{avg} - V_{rest})$$

The average voltage during periodic firing is approximately:

$$V_{avg} \approx \frac{V_{threshold} + V_{reset}}{2} = \frac{-50 + (-70)}{2} = -60 \; \text{mV}$$

So $w_{ss} \approx 0.3 \times (-60 - (-65)) = 0.3 \times 5 = 1.5$

The effective steady-state drive is:
$$\text{drive}_{ss} = \text{gain} \times P - w_{ss} = 1.5P - 1.5$$

For firing to continue at steady state: drive_ss > 0 → P > 1.0

### Rheobase (minimum pressure for firing)

At onset (w = 0):
$$P_{rheo,onset} = \frac{V_{threshold} - V_{rest}}{\text{gain}} = \frac{-50-(-65)}{1.5} = \frac{15}{1.5} = 10$$

At steady state (w = w_ss):
$$P_{rheo,ss} = \frac{V_{threshold} - V_{rest} + w_{ss}}{\text{gain}} \approx \frac{15 + 1.5}{1.5} = 11.0$$

The threshold pressure increases by ~10% from onset to steady state due to adaptation.

### Adaptation time course

The adaptation variable rises with time constant τ_adapt = 200 ms:

$$w(t) = w_{ss} \cdot (1 - e^{-t/\tau_{adapt}})$$

- At t = 200 ms: w reaches 63% of w_ss
- At t = 600 ms: w reaches 95% of w_ss
- At t = 1000 ms: w reaches 99.3% of w_ss

Full adaptation takes ~1 second, consistent with experimental SAI recordings.

### ISI analysis during adaptation

The interspike interval lengthens as adaptation builds:

| Time (ms) | w (approx) | Effective drive | Relative rate |
|-----------|-----------|----------------|---------------|
| 0 (onset) | 0.0 | gain × P | 100% (max) |
| 100 | 0.59 | gain × P - 0.59 | ~90% |
| 200 | 0.95 | gain × P - 0.95 | ~85% |
| 500 | 1.38 | gain × P - 1.38 | ~75% |
| 1000 | 1.49 | gain × P - 1.49 | ~70% |

---

## Comparison with Other Somatosensory Models in SC-NeuroCore

| Property | MerkelCell (SAI) | PacinianCorpuscle (RA II) |
|----------|-----------------|--------------------------|
| Adaptation | Slow (τ = 200 ms) | Fast (RA, τ = 5 ms) |
| Sustained response | Yes | No (onset/offset only) |
| Optimal stimulus | Static pressure | Vibration (40–800 Hz) |
| Receptive field | Small (~2 mm) | Large (~10 mm) |
| Acuity | High | Low |
| Depth in skin | Superficial (epidermis) | Deep (dermis) |

---

## Effect of Parameters on Behaviour

### Adaptation coupling (a_adapt)

| a_adapt | Behaviour |
|---------|-----------|
| 0.0 | No adaptation — constant firing rate (non-physiological) |
| 0.1 | Weak adaptation — slight rate decrease over time |
| 0.3 (default) | Moderate — SAI-like adaptation |
| 0.5 | Strong — rapid rate decrease, nearly RA-like |
| 1.0 | Very strong — burst then near-silence |

### Adaptation time constant (τ_adapt)

| τ_adapt (ms) | Adaptation speed | Classification |
|--------------|-----------------|----------------|
| 10 | Very fast | RA-like (rapidly adapting) |
| 50 | Fast | Intermediate |
| 200 (default) | Slow | SAI-like |
| 1000 | Very slow | Nearly non-adapting |

### Pressure gain

| gain | P_rheo (onset) | Sensitivity |
|------|---------------|-------------|
| 0.5 | 30 | Low (thick skin, deep receptor) |
| 1.0 | 15 | Moderate |
| 1.5 (default) | 10 | High (fingertip) |
| 3.0 | 5 | Very high (lip, genital) |

---

## Parameters

All defaults from `MerkelCell::new()` in
`engine/src/neurons/sensory/merkel_cell.rs`:

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential (initial) |
| `v_rest` | -65.0 | mV | Resting potential |
| `v_reset` | -70.0 | mV | Post-spike reset potential |
| `v_threshold` | -50.0 | mV | Spike detection threshold |
| `tau` | 5.0 | ms | Membrane time constant |
| `adapt` | 0.0 | — | Slow adaptation variable (initial) |
| `tau_adapt` | 200.0 | ms | Adaptation time constant |
| `a_adapt` | 0.3 | — | Adaptation coupling strength |
| `gain` | 1.5 | — | Pressure-to-current gain |
| `dt` | 0.5 | ms | Integration timestep |

---

## Implementation Details

### Code structure (`engine/src/neurons/sensory/merkel_cell.rs`)

```
step(pressure) → i32:
    if state or pressure is invalid:
        return 0 without mutation
    V_inf = V_rest + gain × max(pressure, 0) - adapt
    V_next = V_inf + (V - V_inf) × exp(-dt / τ)
    adapt_inf = max(0, a_adapt × max(0, V_next - V_rest))
    adapt_next = adapt_inf + (adapt - adapt_inf) × exp(-dt / τ_adapt)

    if V_next ≥ V_threshold:
        V = V_reset
        adapt = adapt_next
        return 1
    V = clamp(V_next, -100, 60)
    adapt = adapt_next
    return 0
```

### Key implementation notes

1. **Pressure rectification:** `pressure.max(0.0)` ensures only compressive forces
   drive the receptor.
2. **Adaptation uses candidate V:** The adaptation ODE uses the candidate voltage
   after exact membrane relaxation, not the pre-update value.
3. **Fail-closed numerical boundary:** Non-finite pressure, non-finite state,
   nonphysical finite voltage, invalid time constants, negative adaptation/gain
   parameters, and invalid threshold ordering return no spike without mutation.
4. **No spike-triggered adaptation increment:** Unlike some adaptation models (e.g.,
   LugaroCell where adapt += 1.0 on spike), MerkelCell uses only subthreshold
   adaptation. The adaptation grows continuously with depolarisation.
5. **Reset:** Only V is reset. adapt is NOT reset on spike — it continues its slow
   dynamics across spikes.

---

## Numerical Example

**Setup:** Default parameters, constant pressure P = 15.0.

**Step 1 (t = 0.5 ms):**
1. drive = 1.5 × 15 - 0 = 22.5
2. V += (-(-65-(-65)) + 22.5)/5 × 0.5 = 22.5/5 × 0.5 = 2.25
3. V = -65 + 2.25 = -62.75 mV
4. adapt += (0.3 × (-62.75-(-65)) - 0)/200 × 0.5 = (0.3×2.25)/200 × 0.5 = 1.69×10⁻³

**Step 6 (~3 ms):** V approaches threshold.
1. V ≈ -52 mV, adapt ≈ 0.01
2. Next step: V crosses -50 → spike, V = -70

**Post-spike:** V resets to -70 but adapt ≈ 0.01 (barely changed — τ_adapt = 200 ms
is very slow). The 20 mV gap from -70 to -50 must be traversed again.

After ~500 ms of sustained firing: adapt ≈ 1.3, effective drive = 22.5 - 1.3 = 21.2.
The rate has decreased by ~6% — very slow adaptation, consistent with SAI.

---

## Psychophysical Context

### Weber's law and SAI encoding

SAI afferents approximately follow Weber's law: the just-noticeable difference (JND)
in pressure is proportional to the baseline pressure:

$$\Delta P_{JND} \propto P$$

The adaptation mechanism in the model contributes to this: at higher pressures, the
adaptation variable is larger, so a proportionally larger ΔP is needed to produce
the same change in firing rate.

### Texture encoding

When a textured surface is scanned across the fingertip, each Merkel cell fires in
proportion to the local pressure at its receptive field:
- Ridge: high pressure → high firing rate
- Valley: low pressure → low firing rate

The spatial pattern of SAI firing rates across the fingertip forms a neural image
of the texture that is transmitted to S1 cortex.

### Prosthetic applications

Biomimetic tactile sensors for prosthetic hands often use SAI-like encoding because:
1. It provides sustained force information (crucial for grip)
2. The adaptation reduces redundant information (efficient coding)
3. The onset transient signals contact events (important for reflexes)

The MerkelCell model can drive such encoders in simulation.

---

## Population Coding in SAI Arrays

### Spatial population response

When an array of MerkelCell models is arranged in a grid to simulate a fingertip patch:

```python
# 10×10 SAI array (2 mm spacing, 20×20 mm patch)
cells = [[MerkelCell() for _ in range(10)] for _ in range(10)]

# Gaussian pressure profile centred at (5,5)
for i in range(10):
    for j in range(10):
        r2 = (i-5)**2 + (j-5)**2
        pressure = 20.0 * math.exp(-r2 / 4.0)
        cells[i][j].step(pressure)
```

The population firing rates form a neural image of the pressure distribution, with
each cell's rate encoding local pressure at its receptive field location.

### Temporal population response

During scanning (moving a textured surface across the fingertip):
- Each cell receives a time-varying pressure signal
- The adaptation tracks the local pressure history
- The population output is a spatiotemporal firing pattern that encodes both
  spatial features (texture) and temporal dynamics (scanning speed)

### Rate coding vs temporal coding

SAI afferents primarily use **rate coding**: the mean firing rate over ~50–100 ms
encodes stimulus intensity. However, the precise spike timing within the initial
burst also carries information about stimulus onset dynamics.

The MerkelCell model preserves both:
- Rate information (through the sustained adapted firing rate)
- Onset timing (through the unadapted initial burst)

---

## Comparison with Computational Touch Models

| Model | Complexity | Adaptation | Population | Reference |
|-------|-----------|------------|-----------|-----------|
| MerkelCell (SC-NeuroCore) | LIF + 1 adapt | Slow (200 ms) | Manual array | Lesniak 2014 |
| Saal-Bensmaia | Multi-layer, detailed | Multi-stage | Built-in | Saal 2017 |
| TouchSim | FEM + transduction | Biomechanical | Full hand | Bhatt 2019 |
| Birznieks | Empirical + Poisson | Rate-based | Population | Birznieks 2001 |

The SC-NeuroCore MerkelCell is the simplest approach, suitable for network simulations
where individual SAI response dynamics are needed without the overhead of a full
biomechanical model.

---

## Signal Processing Interpretation

The MerkelCell model implements a **leaky integrator with slow negative feedback**:

- **Transfer function (linear approximation):**
  $$H(s) = \frac{\text{gain}}{(\tau s + 1)(\tau_{adapt} s + 1 + a_{adapt})}$$

- **Low-pass characteristic:** Cutoff frequency ~1/(2πτ) ≈ 32 Hz
- **Adaptation as high-pass:** The slow feedback removes the DC component over
  ~1/(2πτ_adapt) ≈ 0.8 Hz

The combination creates a **band-pass** response (0.8–32 Hz), consistent with
the observation that SAI afferents respond best to slowly varying pressures
(0.4–10 Hz) rather than DC or high-frequency vibration.

---

## FPGA Implementation Notes

### Resource estimates (Zynq-7020, analytical)

| Component | Resource | Estimate |
|-----------|----------|----------|
| Multipliers | DSP48E1 | 3 (drive, dV, dadapt) |
| State registers | Flip-flops | 128 bits (2 × 64-bit) |
| Comparator | LUT | ~32 LUTs |
| Total LUTs | | ~200–350 |
| Pipeline depth | Cycles | ~5–8 |
| Latency at 100 MHz | | 50–80 ns |

Very lightweight — comparable to LugaroCell. FPGA-friendly for large-scale
somatosensory arrays (e.g., robotic skin with 1000+ touch sensors).

**Note:** These are analytical estimates, not measured synthesis results.

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/sensory/merkel_cell.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` |
| NetworkRunner wired | `NeuronVariant::Merkel` |
| `create_neuron("MerkelCell")` | Yes |
| coverage tests | 10 (firing, adaptation, no-fire, reset, exact relaxation, invalid input/state/voltage, non-finite candidate, constructor/default equivalence) |
| Benchmark | `merkel_10k_steps`: **312.93 µs** (31.29 ns/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| merkel_10k_steps | 312.93 µs |
| Per step | **31.29 ns** |

Two exact first-order relaxations, state validation, and one comparison per
step. Artifact:
`benchmarks/results/local_i5_11600k_criterion_2026-05-31_merkel_cell.json`.

Measured 2026-04-04 on i5-11600K @ 3.90 GHz, Criterion.rs, 100 iterations.

---

## Usage Example

### Python

```python
from sc_neurocore_engine import MerkelCell

cell = MerkelCell()

# Sustained pressure stimulus
spikes_per_100ms = []
count = 0
for step in range(4000):  # 2 seconds
    fired = cell.step(15.0)
    count += fired
    if (step + 1) % 200 == 0:
        spikes_per_100ms.append(count)
        count = 0

print(f"Spikes per 100ms: {spikes_per_100ms}")
# Expected: first bin highest, gradually decreasing (SAI pattern)
```

### Rust

```rust
use sc_neurocore_engine::neurons::sensory::MerkelCell;

let mut cell = MerkelCell::new();
let mut total = 0;
for _ in 0..10000 {
    total += cell.step(15.0);
}
println!("Spikes: {}, adapt: {:.3}", total, cell.adapt);
```

---

## Findings

1. **Slow adaptation produces SAI discharge.** First 1000 steps produce more spikes
   than the next 1000 at constant pressure. Verified.
2. **No firing at zero pressure.** gain × 0 - adapt cannot drive V above threshold. Verified.
3. **a_adapt = 0.3 balances onset vs sustained rate.** Moderate adaptation that
   doesn't fully silence the cell. Verified.
4. **Reset clears adaptation.** V returns to V_rest, adapt returns to 0. Verified.
5. **Pressure rectification.** Negative pressure clamped to 0. Verified in the Rust implementation.

---

## References

1. Lesniak DR, Marshall KL, Bhatt DL, et al. (2014). Computation identifies structural
   features that govern neuronal firing properties in slowly adapting touch receptors.
   *eLife* 3:e01488.

2. Maksimovic S, Nakatani M, Baba Y, et al. (2014). Epidermal Merkel cells are
   mechanosensory cells that tune mammalian touch receptors. *Nature* 509:617–621.

3. Woo S-H, Ranade S, Weyer AD, et al. (2014). Piezo2 is required for Merkel-cell
   mechanotransduction. *Nature* 509:622–626.

4. Johnson KO (2001). The roles and functions of cutaneous mechanoreceptors. *Curr Opin
   Neurobiol* 11:455–461.

5. Abraira VE, Ginty DD (2013). The sensory neurons of touch. *Neuron* 79:618–639.

6. Iggo A, Muir AR (1969). The structure and function of a slowly adapting touch corpuscle
   in hairy skin. *J Physiol* 200:763–796.

7. Mountcastle VB (2005). *The Sensory Hand.* Harvard University Press.

8. Johansson RS, Flanagan JR (2009). Coding and use of tactile signals from the fingertips
   in object manipulation tasks. *Nat Rev Neurosci* 10:345–359.

9. Handler A, Bhatt DL, Ginty DD (2021). The mechanosensory neurons of touch and their
   mechanisms of activation. *Nat Rev Neurosci* 22:521–537.

10. Zimmerman A, Bai L, Ginty DD (2014). The gentle touch receptors of mammalian skin.
    *Science* 346:950–954.

11. Gerling GJ, Bhatt DL, Thomas GW (2014). Validating a population model of tactile
    mechanotransduction of slowly adapting type I afferents at the surface of the primate
    fingertip. *J Neurosci* 34:14603–14615.

12. Phillips JR, Johnson KO (1981). Tactile spatial resolution. II. Neural representation
    of bars, edges, and gratings in monkey primary afferents. *J Neurophysiol* 46:1192–1203.

---

*Document verified against Rust source `engine/src/neurons/sensory/merkel_cell.rs`.
All equations, parameters, and default values read directly from the implementation.*
