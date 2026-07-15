# RetinalGanglionCell

**Module:** `engine/src/neurons/sensory/retinal_ganglion_cell.rs`
**Rust struct:** `RetinalGanglionCell`
**Reference:** Pillow et al., Nature 437:1258, 2005; Pillow et al., J Neurosci 28:11003, 2008
**Family:** Generalised Linear Model (GLM) — stimulus filter + history filter + exponential nonlinearity
**State variables:** `stim_buffer` (ring buffer, 20 taps), `hist_buffer` (ring buffer, 30 taps), `stim_idx`, `hist_idx`

---

## Biological Context

Retinal ganglion cells (RGCs) are the spiking output neurons of the retina. Their axons
form the optic nerve (cranial nerve II), carrying all visual information from the eye to
the brain. The human retina contains approximately 1.2 million RGCs, compressing signals
from ~130 million photoreceptors (~100:1 convergence).

### Retinal circuitry

The retinal processing pipeline:

1. **Photoreceptors** (rods, cones): Convert light → graded electrical signals
2. **Horizontal cells**: Lateral inhibition, contrast enhancement
3. **Bipolar cells**: Vertical pathway, ON/OFF separation
4. **Amacrine cells**: Lateral processing, motion detection, adaptation
5. **Retinal ganglion cells**: Final integration → spike trains to brain

RGCs receive excitatory input from bipolar cells and inhibitory input from amacrine
cells. They are the only spiking neurons in the retina — all upstream processing uses
graded potentials.

### RGC types

Approximately 30+ distinct RGC types have been identified in mammals:

| Type | Other names | Receptive field | Function |
|------|------------|-----------------|----------|
| Midget (P) | β, ON/OFF | Small centre-surround | High acuity, colour |
| Parasol (M) | α, ON/OFF | Large centre-surround | Motion, luminance |
| Bistratified (K) | Small bistratified | ON-OFF | Blue-yellow colour |
| Direction-selective | DS-RGC | ON-OFF | Motion direction |
| Intrinsically photosensitive | ipRGC | Very large, sustained | Circadian rhythm |
| Local edge detector | W | Small, suppressed-by-contrast | Edge detection |

The model implements a simplified version applicable to midget or parasol cells with
ON or OFF centre-surround organisation.

### ON and OFF pathways

- **ON-centre cells:** Depolarise (increase firing) when light increases in the centre
  of their receptive field (mediated by ON-bipolar cells via mGluR6)
- **OFF-centre cells:** Depolarise when light decreases (dark increment) in the centre
  (mediated by OFF-bipolar cells via AMPA/kainate receptors)

The ON/OFF distinction is the most fundamental dichotomy in visual processing,
originating at the bipolar cell level and maintained through the entire visual pathway.

### Why a GLM?

The Generalised Linear Model (Pillow et al., 2005) is the gold standard for
statistical modelling of RGC spike trains. It captures three key features:

1. **Temporal filtering:** RGCs have biphasic temporal response — excitation followed
   by suppression (or vice versa). The stimulus filter captures this.
2. **Post-spike history:** RGCs exhibit absolute refractory period, relative
   refractory period, and sometimes burst facilitation. The history filter captures this.
3. **Nonlinear rate coding:** The exponential nonlinearity maps the filtered stimulus
   to an instantaneous firing rate.

---

## Mathematical Model

### Overview

The RetinalGanglionCell implements a **point-process GLM** with three components:

$$\lambda(t) = \exp\!\bigl[k \ast s(t) + h \ast r(t) + b\bigr]$$

where:
- $k \ast s(t)$ is the convolution of the stimulus filter with recent inputs
- $h \ast r(t)$ is the convolution of the history filter with recent spikes
- $b$ is the baseline log-rate
- $\lambda(t)$ is the instantaneous firing rate (Hz)

A spike is emitted deterministically when $\lambda(t) \cdot dt > \theta$.

### Stimulus filter (biphasic temporal kernel)

The stimulus filter has **20 taps** (10 ms history at dt = 0.5 ms). It is constructed
as a difference of Gaussians:

$$k(t) = \frac{\exp\!\left(-\frac{(t-3)^2}{8}\right) - 0.5 \cdot \exp\!\left(-\frac{(t-8)^2}{32}\right)}{\max |k|}$$

where t is the tap index (0–19).

**Excitatory lobe:** Centred at t = 3 (1.5 ms), σ² = 4 → fast initial excitation
**Inhibitory lobe:** Centred at t = 8 (4 ms), σ² = 16, amplitude 0.5 → delayed suppression

The kernel is normalised so the peak absolute value = 1.0.

This biphasic shape is the temporal analogue of the spatial centre-surround receptive
field: it enhances temporal contrast (responds to changes, not steady state).

### Post-spike history filter

The history filter has **30 taps** (15 ms history). It captures post-spike effects:

$$h(t) = -15 \cdot \exp\!\left(-\frac{t}{1.5}\right) + 0.3 \cdot \exp\!\left(-\left(\frac{t-5}{3}\right)^2\right)$$

where t is in ms (tap index × 0.5).

**Refractory component:** $-15 \cdot e^{-t/1.5}$ — strong suppression immediately after
spike, decaying with τ = 1.5 ms. This creates both absolute and relative refractory
periods.

**Burst component:** $0.3 \cdot e^{-((t-5)/3)^2}$ — weak facilitation centred at t = 5 ms
(tap 10), σ = 3 ms. This produces a slight burst tendency: a spike makes the next spike
slightly more likely after ~5 ms.

### ON/OFF polarity

$$I_{eff} = \begin{cases} \text{gain} \times I_{input} & \text{if ON-centre} \\ \text{gain} \times (-I_{input}) & \text{if OFF-centre} \end{cases}$$

The sign flip is the only difference between ON and OFF cells. This matches biology
where the ON/OFF distinction originates from the sign-inverting mGluR6 synapse at
the bipolar cell level.

### Ring buffer convolution

Both stimulus and history are stored in **ring buffers** with circular indexing.
The convolution reads backward from the current write position:

```
convolve(buffer, kernel, write_idx):
    sum = 0
    for i in 0..n:
        buf_idx = (write_idx + n - 1 - i) % n
        sum += buffer[buf_idx] × kernel[i]
    return sum
```

This implements a causal FIR (finite impulse response) filter where kernel[0]
corresponds to the most recent sample and kernel[n-1] to the oldest.

### Exponential nonlinearity

$$\lambda = \min\!\bigl(\exp(\text{filtered\_stim} + \text{filtered\_hist} + b), \; 1000\bigr)$$

The exponential maps the log-rate to an instantaneous firing rate. The cap at 1000 Hz
prevents numerical overflow for very strong inputs.

The baseline $b = -3.0$ sets the spontaneous rate:
$$\lambda_{spont} = e^{-3.0} \times dt = 0.0498 \times 0.5 = 0.0249$$

Since 0.0249 < threshold (0.7), the cell does not fire spontaneously. Spontaneous
firing requires the filtered stimulus to add at least $\ln(0.7/0.5) + 3.0 = 3.34$
to the log-rate.

### Spike decision (deterministic)

$$\text{spike} = \begin{cases} 1 & \text{if } \lambda \cdot dt > \theta \\ 0 & \text{otherwise} \end{cases}$$

where $\theta = 0.7$ (spike threshold). This is a deterministic threshold on the
instantaneous rate, equivalent to the probability of at least one spike in an
inhomogeneous Poisson process when the probability is high.

### Spike history recording

After the spike decision, the result (0 or 1) is written to the history ring buffer.
This creates the feedback loop: past spikes influence future spike probability through
the history filter.

---

## Analytical Properties

### Temporal frequency response

The biphasic stimulus filter acts as a **band-pass** temporal filter:
- Attenuates DC (steady state) — the excitatory and inhibitory lobes partially cancel
- Peaks at ~50–200 Hz (depending on the relative timing of the two lobes)
- Attenuates high frequencies — the Gaussian smoothing limits temporal resolution

This matches the temporal contrast sensitivity of biological RGCs.

### Refractory period analysis

From the history filter:
- At t = 0 ms after spike: h(0) = -15 × 1.0 + 0.3 × e^{-(5/3)²} = -15 + 0.003 ≈ -15
- At t = 1 ms: h(1) = -15 × e^{-0.67} + 0.3 × e^{-(4/3)²} = -7.71 + 0.05 ≈ -7.66
- At t = 3 ms: h(3) = -15 × e^{-2} + 0.3 × e^{-(2/3)²} = -2.03 + 0.19 ≈ -1.84
- At t = 5 ms: h(5) = -15 × e^{-3.33} + 0.3 × e^0 = -0.53 + 0.30 ≈ -0.23
- At t = 8 ms: h(8) = -15 × e^{-5.33} + 0.3 × e^{-1} ≈ -0.07 + 0.11 = +0.04
- At t = 10 ms: h(10) ≈ -0.01 + 0.07 ≈ +0.06

The history filter creates:
- **Absolute refractory** (~0–1 ms): h ≈ -15 to -8, effectively preventing any spike
- **Relative refractory** (~1–5 ms): h ≈ -8 to -0.2, progressively reducing suppression
- **Mild facilitation** (~5–10 ms): h ≈ +0.04, slightly increasing spike probability

### Maximum firing rate

The absolute refractory period of ~1–2 ms limits the maximum firing rate to
approximately 500–1000 Hz. In practice, the relative refractory period limits
sustained firing to ~200–300 Hz, which matches biological RGC maximum rates.

---

## Effect of Parameters on Behaviour

### Baseline (b)

| b | Spontaneous λ×dt | Behaviour |
|---|-----------------|-----------|
| -5.0 | 0.0034 | Very quiet (no spontaneous firing) |
| -3.0 (default) | 0.025 | Low spontaneous rate |
| -1.0 | 0.184 | Moderate spontaneous activity |
| 0.0 | 0.500 | High spontaneous rate |
| +1.0 | 1.359 | Fires almost every step |

### Spike threshold (θ)

| θ | Required λ×dt | Selectivity |
|---|--------------|-------------|
| 0.3 | Low | Very responsive |
| 0.7 (default) | Moderate | Standard |
| 0.9 | High | Selective |

### Gain

| gain | Effect |
|------|--------|
| 0.5 | Reduced stimulus sensitivity |
| 1.0 (default) | Standard |
| 2.0 | Enhanced contrast sensitivity |
| 5.0 | Very high sensitivity (may saturate) |

---

## Parameters

All defaults from `RetinalGanglionCell::new()` in
`engine/src/neurons/sensory/retinal_ganglion_cell.rs`:

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `stim_buffer` | [0.0; 20] | Vec<f64> | Stimulus ring buffer |
| `stim_kernel` | [biphasic; 20] | Vec<f64> | Biphasic temporal filter |
| `stim_idx` | 0 | usize | Stimulus write position |
| `hist_buffer` | [0.0; 30] | Vec<f64> | Spike history ring buffer |
| `hist_kernel` | [refrac+burst; 30] | Vec<f64> | Post-spike history filter |
| `hist_idx` | 0 | usize | History write position |
| `baseline` | -3.0 | — | Baseline log-rate |
| `on_centre` | true | bool | ON-centre (true) / OFF-centre (false) |
| `spike_threshold` | 0.7 | — | λ×dt threshold for spike |
| `dt` | 0.5 | ms | Integration timestep |
| `gain` | 1.0 | — | Input scaling factor |

---

## Implementation Details

### Code structure (`engine/src/neurons/sensory/retinal_ganglion_cell.rs`)

```
step(input) → i32:
    // ON/OFF polarity
    effective = if on_centre: gain × input else: gain × (-input)

    // Write to stimulus ring buffer
    stim_buffer[stim_idx] = effective
    stim_idx = (stim_idx + 1) % 20

    // Convolve stimulus with temporal filter (20 taps)
    filtered_stim = convolve(stim_buffer, stim_kernel, stim_idx)

    // Convolve spike history with post-spike filter (30 taps)
    filtered_hist = convolve(hist_buffer, hist_kernel, hist_idx)

    // GLM: exponential nonlinearity
    log_rate = filtered_stim + filtered_hist + baseline
    λ = min(exp(log_rate), 1000)

    // Deterministic spike decision
    spiked = if λ × dt > spike_threshold: 1 else: 0

    // Record spike in history buffer
    hist_buffer[hist_idx] = spiked as f64
    hist_idx = (hist_idx + 1) % 30

    return spiked
```

### Key implementation notes

1. **GLM, not LIF:** Despite the STUB description, the actual implementation is a
   full Generalized Linear Model with stimulus convolution, history feedback, and
   exponential nonlinearity. There is **no membrane potential V** and **no leak dynamics**.

2. **Ring buffer convolution:** O(n) per call — 20 multiplies for stimulus, 30 for
   history = 50 multiply-adds per step. More expensive than bare LIF but far cheaper
   than conductance-based models.

3. **Rate cap:** λ is capped at 1000 to prevent exp() overflow. With very strong inputs,
   the filtered stimulus can be >10, producing λ = e^10 ≈ 22026, which is capped.

4. **History feedback loop:** The history filter creates a **recurrent** system: spikes
   influence future spike probability. This is the key feature distinguishing the GLM
   from a simple threshold model.

5. **Stimulus kernel construction:** The biphasic kernel is computed in the constructor
   using explicit Gaussian functions, not from empirical data. The normalisation ensures
   peak response = 1.0.

6. **off_centre() constructor:** Creates an OFF-centre cell using struct update syntax:
   `Self { on_centre: false, ..Self::new() }`. All other parameters identical to ON.

7. **reset() clears buffers:** Both ring buffers are zeroed and indices reset to 0.
   This erases all stimulus and spike history.

---

## Numerical Example

**Setup:** Default parameters, step input of 5.0 starting at step 0.

**Step 1:** Input = 5.0 (ON-centre)
1. stim_buffer[0] = 5.0
2. Convolution: only stim_kernel[0] × 5.0 contributes (buffer mostly zero)
   stim_kernel[0] ≈ exp(-(0-3)²/8) - 0.5×exp(-(0-8)²/32) = exp(-1.125) - 0.5×exp(-2.0)
   = 0.325 - 0.068 = 0.257 (before normalisation)
3. filtered_stim ≈ small value (one sample in 20-tap filter)
4. filtered_hist = 0 (no past spikes)
5. log_rate = filtered_stim + 0 + (-3.0) → negative → λ small → no spike

**Steps 1–10:** Stimulus buffer fills up, convolution output grows as the excitatory
lobe of the stimulus filter overlaps with recent stimulus history.

**Step ~5–8:** The biphasic filter's excitatory peak (centred at lag 3) now aligns
with the strongest part of the stimulus onset. filtered_stim is maximised.
If filtered_stim > 3.34 (needed to overcome baseline = -3.0 and reach λ×dt > 0.7):
- log_rate > 0.34 → λ > 1.4 → λ×0.5 = 0.7 → SPIKE

**After first spike:** History filter injects h(0) ≈ -15, strongly suppressing the
next step. The refractory period prevents immediate re-firing.

**Step ~10–15:** Refractory decays, stimulus filter's inhibitory lobe begins to
dominate (centred at lag 8), partially cancelling the excitatory response.
Result: transient burst at stimulus onset, followed by sustained but lower rate.

---

## Coupled GLM Extension

Pillow et al. (2008) extended the basic GLM to a **coupled GLM** where each cell's
spike history influences other cells through cross-coupling filters:

$$\lambda_i(t) = \exp\!\left[k_i \ast s(t) + h_{ii} \ast r_i(t) + \sum_{j \neq i} h_{ij} \ast r_j(t) + b_i\right]$$

The SC-NeuroCore RetinalGanglionCell implements the single-cell version (no cross-coupling).
To simulate a coupled population, multiple cells can share their spike histories manually:

```python
# Pseudo-code for coupled RGC population
cells = [RetinalGanglionCell() for _ in range(10)]
for step in range(N):
    spikes = [cell.step(stim[step]) for cell in cells]
    # Manual cross-coupling would require modifying hist_buffers
```

For production coupled GLM simulations, a dedicated population model would be more
efficient.

---

## Receptive Field Structure

The model implements only the **temporal** component of the receptive field. A complete
spatial-temporal RGC model would include:

**Spatial filter (centre-surround):**
$$S(x, y) = \frac{A_c}{\sigma_c^2} e^{-r^2/2\sigma_c^2} - \frac{A_s}{\sigma_s^2} e^{-r^2/2\sigma_s^2}$$

where r² = x² + y² and σ_s > σ_c (surround wider than centre).

To use with this model, pre-filter the stimulus spatially:
1. Convolve the image with the difference-of-Gaussians spatial filter
2. Pass the filtered value at the cell's location as input to `step()`

---

## Comparison: GLM vs LIF for RGC modelling

| Property | GLM (this model) | LIF |
|----------|-----------------|-----|
| Temporal filtering | Explicit biphasic filter | Implicit (membrane τ) |
| History dependence | Explicit filter (refractory + burst) | Fixed refractory period |
| Nonlinearity | Exponential | Threshold |
| Biological fidelity | Higher (fits real data) | Lower (abstract) |
| Computational cost | 50 multiply-adds/step | 1 multiply-add/step |
| Parameter fitting | Can fit to real spike trains | Manual parameter choice |
| Receptive field | Temporal only (spatial is upstream) | Not modelled |

---

## FPGA Implementation Notes

### Resource estimates (Zynq-7020, analytical)

| Component | Resource | Estimate |
|-----------|----------|----------|
| Multipliers | DSP48E1 | 50 (20 stim + 30 hist convolution) |
| Ring buffers | BRAM | 50 × 64-bit = 400 bytes |
| Exponential | LUT-based | ~100 LUTs |
| Total LUTs | | ~1,500–2,000 |
| Pipeline depth | Cycles | ~30–50 (serialised convolution) |
| Latency at 100 MHz | | 300–500 ns |

**Optimisation:** The 50 multiply-adds can be pipelined or parallelised:
- Fully parallel: 50 DSP slices, 1 cycle
- Serialised: 1 DSP slice, 50 cycles
- Time-multiplexed: useful if simulating many RGCs sharing hardware

**Note:** These are analytical estimates, not measured synthesis results.

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/sensory/retinal_ganglion_cell.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` |
| NetworkRunner wired | `NeuronVariant::RetinalGanglion` |
| `create_neuron("RetinalGanglion")` | Yes |
| coverage tests | 8 (filter contracts, temporal response, ON/OFF firing, no-fire, refractoriness, reset, constructor/default equivalence) |
| Benchmark | `rgc_10k_steps`: **130 µs** (13 ns/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| rgc_10k_steps | 130 µs |
| Per step | **13 ns** |

Despite 50 multiply-adds per step (convolution), the model runs at 13 ns/step because
all data fits in L1 cache (50 taps × 8 bytes = 400 bytes) and the operations are
sequential multiply-accumulate (ideal for CPU pipelines).

Measured 2026-04-04 on i5-11600K @ 3.90 GHz, Criterion.rs, 100 iterations.

---

## Usage Example

### Python

```python
from sc_neurocore_engine import RetinalGanglionCell
import math

# ON-centre RGC
on_cell = RetinalGanglionCell()

# OFF-centre RGC
off_cell = RetinalGanglionCell.off_centre()

# Simulate light flash (step increase in intensity)
on_spikes = []
off_spikes = []
for step in range(2000):
    # Light step at step 200
    intensity = 5.0 if step >= 200 else 0.0
    on_fired = on_cell.step(intensity)
    off_fired = off_cell.step(intensity)
    if on_fired:
        on_spikes.append(step)
    if off_fired:
        off_spikes.append(step)

print(f"ON spikes: {len(on_spikes)} (respond to light onset)")
print(f"OFF spikes: {len(off_spikes)} (respond to light offset/dark)")
```

### Rust

```rust
use sc_neurocore_engine::neurons::sensory::RetinalGanglionCell;

let mut on_cell = RetinalGanglionCell::new();
let mut off_cell = RetinalGanglionCell::off_centre();

let mut on_count = 0;
let mut off_count = 0;

for step in 0..2000 {
    let intensity = if step >= 200 { 5.0 } else { 0.0 };
    on_count += on_cell.step(intensity);
    off_count += off_cell.step(intensity);
}

println!("ON: {}, OFF: {}", on_count, off_count);
```

---

## Findings

1. **ON/OFF polarity via sign flip.** `on_centre` boolean inverts input sign. Matches
   biology (mGluR6 vs AMPA at bipolar synapse). Verified.
2. **Biphasic temporal filter.** Stimulus kernel has fast excitatory lobe (t=3) and
   delayed inhibitory lobe (t=8). Verified from kernel construction code.
3. **Post-spike refractory.** History filter produces strong suppression (-15) at t=0,
   decaying exponentially. Verified from kernel construction.
4. **Mild burst facilitation.** History filter has weak positive lobe at t=5 ms (+0.3).
   Verified from kernel construction.
5. **GLM architecture.** stimulus filter → history filter → exp → threshold. Not
   a simple LIF. Verified from step() implementation.
6. **Rate cap.** λ capped at 1000 to prevent exp overflow. Verified in the Rust implementation.
7. **Reset.** Clears both ring buffers and indices. Verified.

---

## References

1. Pillow JW, Paninski L, Uzzell VJ, et al. (2005). Prediction and decoding of retinal
   ganglion cell responses with a probabilistic spiking model. *J Neurosci* 25:11003–11013.

2. Pillow JW, Shlens J, Paninski L, et al. (2008). Spatio-temporal correlations and visual
   signalling in a complete neuronal population. *Nature* 454:995–999.

3. Chichilnisky EJ (2001). A simple white noise analysis of neuronal light responses.
   *Network* 12:199–213.

4. Paninski L (2004). Maximum likelihood estimation of cascade point-process neural
   encoding models. *Network* 15:243–262.

5. Gollisch T, Meister M (2010). Eye smarter than scientists believed: neural computations
   in circuits of the retina. *Neuron* 65:150–164.

6. Masland RH (2012). The neuronal organisation of the retina. *Neuron* 76:266–280.

7. Baden T, Berens P, Bhatt DL, et al. (2016). The functional diversity of retinal
   ganglion cells in the mouse. *Nature* 529:345–350.

8. Schwartz G, Harris R, Shrom D, et al. (2007). Detection and prediction of periodic
   patterns by the retina. *Nat Neurosci* 10:552–554.

9. Field GD, Chichilnisky EJ (2007). Information processing in the primate retina:
   circuitry and coding. *Annu Rev Neurosci* 30:1–30.

10. Truccolo W, Eden UT, Fellows MR, et al. (2005). A point process framework for relating
    neural spiking activity to spiking history, neural ensemble, and extrinsic covariate
    effects. *J Neurophysiol* 93:1074–1089.

11. Keat J, Reinagel P, Reid RC, et al. (2001). Predicting every spike: a model for the
    responses of visual neurons. *Neuron* 30:803–817.

12. Berry MJ, Warland DK, Meister M (1997). The structure and precision of retinal spike
    trains. *PNAS* 94:5411–5416.

---

*Document verified against Rust source `engine/src/neurons/sensory/retinal_ganglion_cell.rs`.
All equations, filter construction, and default values read directly from the
implementation. The STUB incorrectly described this as a simple LIF with refractory
period — the actual implementation is a full GLM with convolution filters.*
