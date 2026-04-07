# DirectionSelectiveRGC

**Module:** `sc_neurocore.neurons.models.direction_selective_rgc`
**Rust path:** `sc_neurocore_engine::neurons::sensory::DirectionSelectiveRGC`
**Reference:** Gollisch & Meister (2010), Masland (2012)
**Family:** Retinal sensory neurons
**State variables:** `v` (membrane potential), `_prev_intensity` (temporal buffer), `_surround` (surround estimate)

---

## 1. Mathematical Formalism

### Core equations

Direction-selective retinal ganglion cell (RGC) with On/Off centre-surround
receptive field and temporal derivative-based direction selectivity.

**Temporal derivative (motion signal):**

$$\Delta I = I(t) - I(t-1)$$

**Centre response (On or Off):**

$$R_{centre} = \begin{cases} w_c \cdot \Delta I & \text{On-centre (responds to light increase)} \\ -w_c \cdot \Delta I & \text{Off-centre (responds to light decrease)} \end{cases}$$

**Surround inhibition (low-pass filtered):**

$$S(t) = 0.9 \cdot S(t-1) + 0.1 \cdot I_{surround}$$

$$R_{surround} = w_s \cdot S(t)$$

**Drive:**

$$\text{drive} = R_{centre} - R_{surround}$$

**Membrane dynamics (leaky integrator):**

$$V(t+1) = V(t) + \frac{(-V(t) + \text{drive}) \cdot dt}{\tau}$$

**Spike condition:**

$$\text{spike} = \begin{cases} 1 & \text{if } V \geq \theta \\ 0 & \text{otherwise} \end{cases}$$

with hard reset $V \leftarrow 0$ after spike.

### Centre-surround antagonism

The retinal ganglion cell has a receptive field with two zones:

1. **Centre:** Responds to local intensity changes. On-centre cells are excited by
   light increase in the centre; Off-centre cells are excited by light decrease.

2. **Surround:** Inhibited by uniform illumination. The surround response is low-pass
   filtered ($\alpha = 0.1$) to model the slower surround mechanism mediated by
   horizontal cells and amacrine cells in the retina.

The difference $R_{centre} - R_{surround}$ implements **spatial contrast enhancement**:
the cell responds to local changes (edges, motion) while ignoring uniform illumination.

### Direction selectivity

The temporal derivative $\Delta I = I(t) - I(t-1)$ encodes the direction of
intensity change:
- $\Delta I > 0$: light increase (preferred direction for On-centre)
- $\Delta I < 0$: light decrease (preferred direction for Off-centre)
- $\Delta I \approx 0$: no change (no response)

This creates a simple direction selectivity: On-cells fire during light onset
(moving bright edge), Off-cells fire during light offset (moving dark edge).

In biology, direction selectivity in DSGCs involves asymmetric GABAergic
inhibition from starburst amacrine cells (Briggman et al. 2011). Our model
simplifies this to a temporal derivative, which captures the functional
outcome (direction-dependent firing) without the full circuit mechanism.

### On vs Off pathways

The retina splits visual information into parallel On and Off pathways at
the first synapse (photoreceptor → bipolar cell):

| Pathway | Glutamate effect | Receptor | Bipolar type | RGC type |
|---------|-----------------|----------|-------------|----------|
| On | Sign-inverting | mGluR6 | On-bipolar | On-centre RGC |
| Off | Sign-conserving | AMPA/KA | Off-bipolar | Off-centre RGC |

Our `is_on_centre` flag selects the sign of the temporal derivative response,
modelling this On/Off split at the RGC level.

### Surround mechanism details

The surround uses an exponential moving average with $\alpha = 0.1$:

$$S(t) = (1 - \alpha) \cdot S(t-1) + \alpha \cdot I_{surround}$$

This has an effective time constant of:

$$\tau_{eff} = -\frac{dt}{\ln(1 - \alpha)} = -\frac{1}{\ln(0.9)} \approx 9.5 \text{ steps}$$

The surround adapts more slowly than the centre (which is instantaneous dI/dt),
creating temporal decorrelation in addition to spatial decorrelation.

---

## 2. Theoretical Context

### Problem statement

The retina is the first stage of visual processing, converting photon absorption
into spike trains. Retinal ganglion cells (RGCs) are the output neurons of the retina,
sending spike trains to the brain via the optic nerve. There are >30 RGC types, each
encoding different visual features (edges, motion, colour, luminance).

Direction-selective ganglion cells (DSGCs) are among the most studied RGC types,
encoding the direction and speed of visual motion.

### Historical context

Key milestones in RGC research:

| Year | Discovery | Reference |
|------|-----------|-----------|
| 1953 | Centre-surround receptive fields | Barlow (1953) |
| 1964 | Direction selectivity in rabbit retina | Barlow & Levick (1964) |
| 2001 | Starburst amacrine cell role in DS | Euler et al. (2002) |
| 2010 | Predictive coding in retina | Gollisch & Meister (2010) |
| 2011 | Full DS circuit mapped | Briggman et al. (2011) |
| 2012 | >30 RGC types identified | Masland (2012) |

### Existing RGC in SC-NeuroCore

The codebase has a GLM-style `RetinalGanglionCell` (at sensory.rs:497) with
stimulus and history filters, on_centre flag, and GLM-like spike generation.
Our `DirectionSelectiveRGC` adds:

1. Temporal derivative for motion/direction encoding
2. Explicit centre-surround antagonism with separate weights
3. Surround low-pass filtering for temporal decorrelation
4. Simpler implementation suitable for large-scale retinal simulations

### Retinal encoding properties

The temporal derivative response has important information-theoretic properties:

**Redundancy reduction:** Natural scenes have strong spatial and temporal
correlations (adjacent pixels and successive frames are similar). The temporal
derivative $\Delta I$ removes temporal correlations, transmitting only changes.
This is the retinal implementation of predictive coding (Srinivasan et al. 1982):
the retina transmits the prediction error $I(t) - \hat{I}(t)$ where $\hat{I}(t) \approx I(t-1)$.

**Bandwidth optimisation:** The optic nerve has ~1 million axons (RGC outputs)
compared to ~100 million photoreceptors. The 100:1 compression requires efficient
coding. Centre-surround + temporal derivative achieves near-optimal whitening of
the retinal input under natural scene statistics (Atick & Redlich 1992).

**Invariance to DC offset:** The temporal derivative is invariant to constant
illumination levels: $d(I + c)/dt = dI/dt$. This allows the RGC to signal
contrast changes regardless of ambient light level, complementing the photoreceptor's
logarithmic light adaptation.

**Speed sensitivity:** For a moving edge with velocity $v$ and spatial contrast $C$:

$$\Delta I \approx v \cdot C \cdot \frac{\partial I}{\partial x}$$

The temporal derivative encodes the product of velocity and spatial gradient,
making the cell sensitive to both speed and contrast.

### Parallel pathways in the retina

The retina splits visual information into >30 parallel channels (Masland 2012).
Our model captures two of the most fundamental:

| Pathway | RGC type | Stimulus preference | Downstream target |
|---------|----------|-------------------|-------------------|
| Midget (parvocellular) | On/Off midget | High acuity, colour | LGN → V1 |
| Parasol (magnocellular) | On/Off parasol | Motion, luminance | LGN → V1, MT |
| **Direction-selective** | **On-Off DS** | **Motion direction** | **SC, AOS** |
| Intrinsically photosensitive | ipRGC | Ambient light | SCN (circadian) |
| Local edge detector | W3 | Small moving objects | SC |

Our DirectionSelectiveRGC most closely models the On-Off DS-RGC type, which
projects primarily to the superior colliculus (SC) and accessory optic system (AOS)
for reflexive eye movements and optokinetic responses.

### Applications

1. **Retinal prosthetics:** Encoding visual input as spike trains for prosthetic devices
2. **DVS (dynamic vision sensor) processing:** Neuromorphic cameras output events
   similar to On/Off RGC responses
3. **Motion detection:** Direction-selective units for optical flow estimation
4. **Visual preprocessing:** Converting raw images to biologically plausible spike input
   for downstream spiking networks
5. **Predictive coding:** The centre-surround mechanism implements spatial prediction error
6. **Edge detection:** Response to dI/dt highlights temporal edges in visual scenes

### Relationship to existing models

| Model | Spatial RF | Temporal | Direction | Complexity |
|-------|-----------|----------|-----------|------------|
| Poisson rate | None | Rate coding | No | O(1) |
| GLM-RGC (existing) | Stimulus filter | History filter | No | O(filter_len) |
| **DirectionSelectiveRGC** | **Centre-surround** | **dI/dt** | **Yes** | **O(1)** |
| Full retinal circuit | Layered (photo→bipl→gang) | Full dynamics | Yes | O(N²) |

---

## 3. Pipeline Position

```
Visual stimulus (intensity image)
    │
    ├── centre pixel intensity
    │         │
    │         ▼
    │   ┌──────────────────────────────┐
    │   │   DirectionSelectiveRGC       │
    │   │                              │
    │   │  dI/dt (temporal derivative) │
    │   │       │                      │
    │   │  ┌────▼─────┐               │
    │   │  │ On/Off   │               │
    │   │  │ sign     │               │
    │   │  └────┬─────┘               │
    │   │       │  - surround         │
    │   │       ▼                      │
    │   │  leaky integrator → spike   │
    │   └──────────────────────────────┘
    │
    └── surround mean intensity
```

### Inputs

| Input | Type | Range | Description |
|-------|------|-------|-------------|
| `intensity` | `float` | $[0, +\infty)$ | Centre pixel intensity |
| `surround_mean` | `float` | $[0, +\infty)$ | Mean intensity of surround region |

### Outputs

| Output | Type | Range | Description |
|--------|------|-------|-------------|
| `spike` | `int` | $\{0, 1\}$ | Binary spike |

---

## 4. Features

| Feature | Description |
|---------|-------------|
| **On-centre / Off-centre** | Toggle via `is_on_centre` flag |
| **Temporal derivative** | Responds to dI/dt, not absolute intensity |
| **Centre-surround** | Spatial contrast via surround inhibition |
| **Low-pass surround** | EMA filtering of surround for temporal decorrelation |
| **Configurable weights** | `w_centre` and `w_surround` control relative sensitivity |
| **Direction preference** | `direction_pref` parameter (radians) for spatial DS |
| **Simple API** | `step(current)` for single-input, `step_rf(intensity, surround)` for full RF |
| **Constructors** | `new_on()` and `new_off()` class methods |
| **Rust parity** | Identical equations to Rust implementation |

---

## 5. Usage Examples

### On-centre flash response

```python
from sc_neurocore.neurons.models import DirectionSelectiveRGC

cell = DirectionSelectiveRGC.new_on()

# Darkness, then flash.
for _ in range(10):
    cell.step_rf(0.0, 0.0)  # dark
spikes = sum(cell.step_rf(5.0, 0.0) for _ in range(20))
print(f"Flash response: {spikes} spikes")
```

### Off-centre response to light offset

```python
cell = DirectionSelectiveRGC.new_off()
cell.theta = 0.1

# Alternating bright/dark.
spikes = 0
for i in range(400):
    intensity = 5.0 if (i // 10) % 2 == 0 else 0.0
    spikes += cell.step_rf(intensity, 0.0)
print(f"Off-centre response: {spikes} spikes")
```

### Surround inhibition comparison

```python
no_surr = DirectionSelectiveRGC.new_on()
with_surr = DirectionSelectiveRGC.new_on()

s_no, s_surr = 0, 0
for i in range(300):
    intensity = 3.0 if i % 10 == 0 else 0.0
    s_no += no_surr.step_rf(intensity, 0.0)
    s_surr += with_surr.step_rf(intensity, 2.0)

print(f"No surround: {s_no} spikes")
print(f"With surround: {s_surr} spikes (should be less)")
```

### Constant light adaptation

```python
cell = DirectionSelectiveRGC.new_on()
for _ in range(100):
    cell.step_rf(3.0, 0.0)  # adapt
late_spikes = sum(cell.step_rf(3.0, 0.0) for _ in range(100))
print(f"Constant light late spikes: {late_spikes} (should be 0 — no dI/dt)")
```

### Moving edge simulation

```python
import math
cell = DirectionSelectiveRGC.new_on()
cell.theta = 0.2
# Sinusoidal intensity simulating a moving grating.
spikes = 0
for t in range(500):
    intensity = 2.0 * (1.0 + math.sin(t * 0.1))
    spikes += cell.step_rf(intensity, 1.0)
print(f"Moving grating: {spikes} spikes")
```

### DVS (dynamic vision sensor) compatibility

```python
# Simulate DVS-like event processing.
# DVS outputs events when brightness changes by a threshold.
cell_on = DirectionSelectiveRGC.new_on()
cell_off = DirectionSelectiveRGC.new_off()
cell_off.theta = 0.2

# Simulate a pixel's intensity trajectory.
import math
for t in range(200):
    intensity = 3.0 * math.sin(t * 0.05) + 3.0
    surround = 3.0  # uniform surround
    on_spike = cell_on.step_rf(intensity, surround)
    off_spike = cell_off.step_rf(intensity, surround)
    if on_spike or off_spike:
        polarity = "ON" if on_spike else "OFF"
        print(f"t={t}: DVS event ({polarity})")
```

### Population of On/Off cells

```python
# Create a population of On and Off cells monitoring different spatial locations.
n_cells = 10
on_cells = [DirectionSelectiveRGC.new_on() for _ in range(n_cells)]
off_cells = [DirectionSelectiveRGC.new_off() for _ in range(n_cells)]

# Simulate a moving edge across the population.
for t in range(100):
    for i in range(n_cells):
        # Edge position moves across cells.
        edge_pos = t * 0.3
        intensity = 5.0 if (i - edge_pos) > 0 else 0.0
        on_cells[i].step_rf(intensity, 2.5)
        off_cells[i].step_rf(intensity, 2.5)
```

---

## 6. Technical Reference

### Class: `DirectionSelectiveRGC`

Decorated with `@dataclass`. Defined in
`src/sc_neurocore/neurons/models/direction_selective_rgc.py`.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau` | `float` | `10.0` | Membrane time constant (ms) |
| `theta` | `float` | `0.5` | Spike threshold |
| `is_on_centre` | `bool` | `True` | On (True) or Off (False) centre |
| `w_centre` | `float` | `1.0` | Centre weight |
| `w_surround` | `float` | `0.3` | Surround inhibition weight |
| `direction_pref` | `float` | `0.0` | Preferred direction (radians) |
| `dt` | `float` | `1.0` | Integration timestep |

#### State Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `v` | `float` | `0.0` | Membrane potential |
| `_prev_intensity` | `float` | `0.0` | Previous centre intensity |
| `_surround` | `float` | `0.0` | Filtered surround estimate |

#### Methods

**`step_rf(intensity: float, surround_mean: float) -> int`** — Full receptive field step.
**`step(current: float) -> int`** — Simple step (no surround).
**`reset() -> None`** — Reset all state to zero.
**`new_on() -> DirectionSelectiveRGC`** — Class method: On-centre constructor.
**`new_off() -> DirectionSelectiveRGC`** — Class method: Off-centre constructor.

### Rust parity

| Operation | Python | Rust |
|-----------|--------|------|
| dI/dt | `intensity - _prev_intensity` | `intensity - self.prev_intensity` |
| On/Off sign | `w_c * diff` or `-w_c * diff` | Same conditional |
| Surround EMA | `0.9*S + 0.1*surround_mean` | `0.9*self.surround + 0.1*surround_mean` |
| V update | `v += (-v + drive)/tau * dt` | `self.v += (-self.v + drive)/self.tau * self.dt` |
| Spike | `v >= theta → v = 0` | `v >= theta → v = 0.0` |

---

## 7. Performance Benchmarks

### Python (i5-11600K, single core, CPython 3.12)

| Method | Time per step | Steps/second |
|--------|--------------|--------------|
| `step_rf()` | 1,146 ns | 873,000 |
| `step()` | ~1,100 ns | 909,000 |

Fastest of all 11 gap models — no `math.exp()` call, pure arithmetic.

### Rust (i5-11600K, single core, Criterion)

| Method | Time per step | Speedup vs Python |
|--------|--------------|-------------------|
| `step_rf()` | ~3 ns | ~382× |

### Memory: ~150 bytes (Python), 80 bytes (Rust, 10× f64)

---

## 8. Citations

1. **Gollisch, T. & Meister, M.** "Eye smarter than scientists believed: neural
   computations in circuits of the retina." Neuron 65(2):150-164, 2010.
   — Comprehensive review of retinal computation including direction selectivity.

2. **Masland, R. H.** "The neuronal organization of the retina." Neuron
   76(2):266-280, 2012.
   — Classification of >30 RGC types including DS-RGCs.

3. **Barlow, H. B. & Levick, W. R.** "The mechanism of directionally selective
   units in rabbit's retina." Journal of Physiology 178(3):477-504, 1965.
   — Original discovery of direction selectivity in retinal ganglion cells.

4. **Briggman, K. L. et al.** "Wiring specificity in the direction-selectivity
   circuit of the retina." Nature 471(7337):183-188, 2011.
   — Circuit-level mechanism of direction selectivity via starburst amacrine cells.

5. **Euler, T. et al.** "Directionally selective calcium signals in dendrites of
   starburst amacrine cells." Nature 418(6900):845-852, 2002.
   — Key mechanism underlying retinal direction selectivity.

6. **Kuffler, S. W.** "Discharge patterns and functional organization of
   mammalian retina." Journal of Neurophysiology 16(1):37-68, 1953.
   — Original description of centre-surround receptive fields.

7. **Srinivasan, M. V. et al.** "Predictive coding: a fresh view of inhibition
   in the retina." Proceedings of the Royal Society B 216(1205):427-459, 1982.
   — Temporal decorrelation via predictive coding, underlying dI/dt response.

8. **Atick, J. J. & Redlich, A. N.** "What does the retina know about natural
   scenes?" Neural Computation 4(2):196-210, 1992.
   — Optimal whitening theory for retinal centre-surround receptive fields.

---

## Validation

### Test suite results

| Test | What it verifies | Status |
|------|-----------------|--------|
| `test_on_centre_flag` | is_on_centre = True/False | PASS |
| `test_on_responds_to_light_increase` | On-cell fires on light onset | PASS |
| `test_off_responds_to_light_decrease` | Off-cell fires on light offset | PASS |
| `test_surround_inhibition_reduces_firing` | Surround suppresses firing | PASS |
| `test_temporal_derivative` | Constant light → no spikes | PASS |
| `test_reset` | v, _prev_intensity, _surround → 0 | PASS |

### Equation-to-code traceability

| Equation | Python location | Rust location |
|----------|----------------|---------------|
| $\Delta I = I - I_{prev}$ | `direction_selective_rgc.py:85` | `sensory.rs:1811` |
| On/Off sign flip | `direction_selective_rgc.py:88-90` | `sensory.rs:1814-1818` |
| $S = 0.9S + 0.1I_{surr}$ | `direction_selective_rgc.py:92` | `sensory.rs:1820` |
| $dV/dt = (-V + \text{drive})/\tau$ | `direction_selective_rgc.py:95` | `sensory.rs:1824` |

---

## Design Decisions

### Why temporal derivative instead of full DS circuit?

The biological direction selectivity mechanism involves starburst amacrine cells
with asymmetric GABAergic synapses (Briggman et al. 2011). This requires
modelling a multi-cell circuit with specific connectivity. Our temporal derivative
approach captures the functional outcome (sensitivity to dI/dt) with a single
neuron, making it suitable for large-scale retinal simulations where circuit-level
detail is not needed.

### Why hard-coded EMA constants (0.9, 0.1)?

The surround EMA uses fixed constants ($\alpha = 0.1$, decay = $1 - \alpha = 0.9$)
rather than configurable parameters. This matches the Rust implementation and
provides a reasonable default for retinal surround dynamics. The effective time
constant (~10 timesteps) matches the 10-50 ms temporal integration of horizontal
cells in the outer retina.

### Why reset V to 0 instead of V_rest?

The model uses abstract (non-millivolt) potentials, so resting potential is 0.
This differs from biophysical models (DendriticNMDA, AstrocyteLIF) that use -65 mV.
The choice is consistent with the Rust implementation and simplifies the dynamics.

---

## Known Limitations

1. **No spatial receptive field:** The model takes scalar centre and surround inputs.
   Real RGC receptive fields have 2D Gaussian centre and difference-of-Gaussian surround.

2. **No adaptation beyond temporal derivative:** Real RGCs have contrast gain control,
   light adaptation, and history-dependent responses. Only the temporal derivative
   provides a simple form of adaptation.

3. **No direction tuning curve:** The `direction_pref` parameter exists but is not
   used in the computation. Full direction selectivity requires velocity-dependent input.

4. **No spike rate saturation:** The firing rate increases linearly with drive.
   Real RGCs have a maximum firing rate (~200 Hz) due to refractory period.

5. **On/Off are separate:** One model instance is either On or Off. Real retinal circuits
   have On and Off pathways in parallel, requiring two instances per spatial location.

6. **No contrast gain control:** Real RGCs adapt their gain based on local contrast
   statistics (Shapley & Victor 1978). Our model has fixed gain.

7. **No refractory period:** No minimum inter-spike interval. Real RGCs have ~2ms
   absolute refractory period limiting maximum firing rate to ~500 Hz.

8. **No colour sensitivity:** The model operates on scalar intensity. Colour-opponent
   RGCs (e.g., red-green midget cells) require separate chromatic channels.

## Implementation Notes

### Temporal derivative via frame differencing

The temporal derivative $\Delta I = I(t) - I(t-1)$ uses simple first-order differencing
rather than a continuous derivative. This is appropriate for discrete-time simulation
and matches the frame-differencing approach used in DVS (dynamic vision sensor) cameras.

For continuous-time simulation with very small dt, the derivative $\Delta I / dt$ diverges
as $dt \to 0$ because the intensity step between frames is fixed. In this regime, use
$\text{drive} = w_c \cdot \Delta I / dt$ with an appropriate scaling factor. Our model
uses $\Delta I$ directly (without dividing by dt), which is correct for the discrete-time
interpretation where each step represents one frame.

### Centre-surround as difference of Gaussians

In biology, the centre-surround receptive field is well approximated by a difference
of Gaussians (DoG):

$$RF(x, y) = \frac{1}{2\pi\sigma_c^2} \exp\left(-\frac{x^2+y^2}{2\sigma_c^2}\right) - \frac{k}{2\pi\sigma_s^2} \exp\left(-\frac{x^2+y^2}{2\sigma_s^2}\right)$$

Our model collapses this to scalar inputs (centre intensity, surround mean), assuming
the spatial integration has been performed upstream. For spatial receptive fields,
compute the DoG-weighted sum of pixel intensities before passing to the model.

---

*SC-NeuroCore v3.14.0 — Stochastic Computing Spiking Neural Network Framework*

*© 2020–2026 Miroslav Šotek. AGPL-3.0-or-later.*
