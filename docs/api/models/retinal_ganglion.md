# RetinalGanglionCell

**Module:** `engine/src/neurons/sensory.rs`
**Reference:** Pillow et al. 2005 (GLM framework, simplified)
**Family:** Spiking sensory output neuron, leaky integrate-and-fire with refractory period
**State variables:** `v` (membrane potential), `refrac_count` (refractory counter)

---

## Biological Context

Retinal ganglion cells (RGCs) are the spiking output neurons of the retina. Their axons form the optic nerve, carrying all visual information to the brain. Roughly 1.2 million RGCs per human eye compress ~130 million photoreceptor signals.

Key features:
- Receive graded input from bipolar cells (which integrate photoreceptor signals)
- ON-centre cells: depolarise (fire) in response to light increment in centre of receptive field
- OFF-centre cells: depolarise (fire) in response to light decrement (dark) in centre
- Produce conventional action potentials with refractory period
- Firing rate encodes stimulus contrast, not absolute luminance (contrast gain control)

The model implements a LIF neuron with ON/OFF polarity selection and a discrete refractory period. The `on_centre` flag inverts the sign of bipolar cell input for OFF-centre cells.

---

## Equations

### Membrane dynamics (LIF)

$$\frac{dV}{dt} = \frac{-(V - V_{rest}) + gain \cdot I_{eff}}{\tau}$$

where the effective input depends on cell polarity:

$$I_{eff} = \begin{cases} I_{input} & \text{if ON-centre} \\ -I_{input} & \text{if OFF-centre} \end{cases}$$

### Spike and reset

$$\text{if } V \geq V_{threshold}: \quad V \leftarrow V_{reset}, \quad refrac\_count \leftarrow refrac\_period, \quad \text{emit spike (1)}$$

### Refractory period

During refractory ($refrac\_count > 0$): no integration, no spikes, counter decrements each step. Returns 0.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `v_rest` | -65.0 | mV | Resting potential |
| `v_reset` | -70.0 | mV | Post-spike reset potential |
| `v_threshold` | -50.0 | mV | Spike threshold |
| `tau` | 10.0 | ms | Membrane time constant |
| `on_centre` | true | — | ON-centre (true) or OFF-centre (false) |
| `gain` | 2.0 | — | Contrast gain factor |
| `refrac_count` | 0 | steps | Current refractory counter |
| `refrac_period` | 3 | steps | Refractory period duration |
| `dt` | 0.5 | ms | Integration timestep |

Alternative constructor: `RetinalGanglionCell::off_centre()` creates an OFF-centre cell (all other parameters identical).

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/sensory.rs` |
| PyO3 wrapper | `py_neuron_default!` macro in `pyo3_neurons.rs` |
| NetworkRunner wired | `NeuronVariant::RetinalGanglion` |
| `create_neuron("RetinalGanglion")` | Yes |
| STRONG tests | 5 (ON fires, OFF fires, no-fire, refractory period, reset) |
| NaN/extreme input test | Via NetworkRunner `all_models_*` tests |
| Benchmark | `rgc_10k_steps`: **130 µs** (13 ns/step), i5-11600K |

---

## Benchmark (i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| rgc_10k_steps | 130 µs |
| Per step | **13 ns** |

The step function is a simple LIF: one comparison, one linear ODE, no transcendental functions. Expected cost in the low nanosecond range per step.

---

## Findings

1. **ON/OFF polarity is a sign flip, not a separate model.** The `on_centre` boolean inverts input sign. This matches the biological observation that ON and OFF pathways differ primarily in the sign of bipolar cell glutamate response (mGluR6 vs AMPA).
2. **Refractory period enforced.** The `rgc_refractory_period` test verifies that no spike can occur in the step immediately following a spike.
3. **gain = 2.0 provides contrast amplification.** Bipolar cell input is doubled before integration, reflecting the retinal gain control that enhances contrast sensitivity.
4. **OFF-centre fires with negative input.** The sign inversion means the same magnitude of negative (dark) input drives OFF cells as positive (light) input drives ON cells.
