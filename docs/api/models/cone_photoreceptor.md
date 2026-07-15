# ConePhotoreceptor

**Module:** `engine/src/neurons/sensory/cone_photoreceptor.rs`
**Reference:** Schnapf et al. 1990; Baylor 1987
**Family:** Graded sensory receptor, photopic phototransduction
**State variables:** `v` (membrane potential), `cgmp` (normalised cGMP concentration)

---

## Biological Context

Cone photoreceptors mediate photopic (bright light) colour vision. Roughly 6 million cones per human retina (concentrated in the fovea) provide colour discrimination and high temporal resolution at the cost of lower absolute sensitivity than rods.

Key features:
- Same transduction cascade as rods (rhodopsin/opsin -> transducin -> PDE -> cGMP -> CNG), but with faster kinetics at every stage
- Lower sensitivity (higher light threshold) than rods
- Much faster dark adaptation: tau_rec = 50 ms vs 500 ms for rods
- Faster activation: tau_act = 5 ms vs 20 ms for rods
- Smaller hyperpolarisation range: V_hyper = -65 mV vs -70 mV for rods
- Graded output, no action potentials

The model uses the same equations as RodPhotoreceptor with different parameter values reflecting the faster, less sensitive cone cascade.

---

## Equations

### cGMP dynamics

$$\frac{d[cGMP]}{dt} = -\frac{S \cdot I \cdot [cGMP]}{\tau_{act}} + \frac{1 - [cGMP]}{\tau_{rec}}$$

where $S$ is sensitivity, $I$ is light intensity (clamped $\geq 0$). cGMP is clamped to $[0, 1]$.

### CNG channel fraction

$$f_{CNG} = [cGMP]^3$$

### Membrane potential

$$V = V_{hyper} + (V_{dark} - V_{hyper}) \cdot f_{CNG}$$

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -40.0 | mV | Membrane potential |
| `v_dark` | -40.0 | mV | Dark resting potential |
| `v_hyper` | -65.0 | mV | Maximum hyperpolarised potential |
| `cgmp` | 1.0 | — | Normalised cGMP concentration [0, 1] |
| `tau_act` | 5.0 | ms | Activation time constant |
| `tau_rec` | 50.0 | ms | Recovery time constant |
| `sensitivity` | 0.001 | — | Light-to-PDE coupling gain |
| `dt` | 0.1 | ms | Integration timestep |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/sensory/cone_photoreceptor.rs` |
| PyO3 wrapper | `py_sensory_graded!` macro in `pyo3_neurons.rs` |
| NetworkRunner wired | **No** — graded model, `step()` returns `f64` |
| `create_neuron("ConePhotoreceptor")` | No (not in NetworkRunner variant enum) |
| coverage tests | 4 (hyperpolarise, faster-than-rod comparative, reset, constructor/default equivalence) |
| NaN/extreme input test | Inherits rod's cGMP clamping logic |
| Benchmark | Not benchmarked (shares arch with rod, ~30 ns/step) |

Graded sensory models are accessed directly via PyO3 (`ConePhotoreceptor` class) or Rust. They are not routed through `NetworkRunner` because the network step loop expects `i32` spike outputs.

---

## Benchmark (i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| 10k steps | ~308 µs (estimated) |
| Per step | ~30 ns |

Identical computation to RodPhotoreceptor (one `powi(3)`, one `clamp`, arithmetic). Expected cost under 500 ns/step.

---

## Comparison with RodPhotoreceptor

| Property | Rod | Cone (this) |
|----------|-----|-------------|
| `sensitivity` | 0.01 | 0.001 (10x lower) |
| `tau_act` | 20.0 ms | 5.0 ms (4x faster) |
| `tau_rec` | 500.0 ms | 50.0 ms (10x faster) |
| `v_hyper` | -70.0 mV | -65.0 mV (smaller range) |
| Modality | Scotopic | Photopic |

---

## Findings

1. **10x faster recovery than rods.** tau_rec = 50 ms allows cones to follow flicker at frequencies where rods are still recovering.
2. **10x lower sensitivity.** Cones require stronger light to produce the same cGMP hydrolysis rate, matching their higher absolute threshold.
3. **Comparative test validates kinetic ordering.** The `cone_faster_than_rod` test confirms cones recover more after a matched light flash followed by dark period.
4. **Reset restores cGMP to 1.0 and V to V_dark.** Clean state recovery.
