# RodPhotoreceptor

**Module:** `engine/src/neurons/sensory.rs`
**Reference:** Nikonov et al. 2006; Hamer et al. 2005
**Family:** Graded sensory receptor, scotopic phototransduction
**State variables:** `v` (membrane potential), `cgmp` (normalised cGMP concentration)

---

## Biological Context

Rod photoreceptors mediate scotopic (dim light) vision. Roughly 120 million rods per human retina provide single-photon sensitivity at the cost of slow temporal dynamics and no colour discrimination.

Key features:
- In darkness: high cGMP keeps CNG channels open, sustaining a "dark current" that depolarises the cell to ~-40 mV
- Light activates rhodopsin -> transducin -> PDE, which hydrolyses cGMP
- cGMP drop closes CNG channels -> hyperpolarisation (the "light response")
- CNG current is proportional to [cGMP]^3 (Hill coefficient ~3)
- Recovery requires cGMP resynthesis by guanylyl cyclase, regulated by GCAP/Ca2+ feedback
- Very slow dark adaptation: tau_rec = 500 ms in this model
- Graded output, no action potentials

The model captures the cGMP cascade with first-order kinetics for activation and recovery, and the CNG Hill nonlinearity.

---

## Equations

### cGMP dynamics

$$\frac{d[cGMP]}{dt} = -\frac{S \cdot I \cdot [cGMP]}{\tau_{act}} + \frac{1 - [cGMP]}{\tau_{rec}}$$

where $S$ is sensitivity, $I$ is light intensity (clamped $\geq 0$), $\tau_{act}$ is the activation time constant, and $\tau_{rec}$ is the recovery time constant. cGMP is clamped to $[0, 1]$.

### CNG channel fraction (Hill function)

$$f_{CNG} = [cGMP]^3$$

### Membrane potential

$$V = V_{hyper} + (V_{dark} - V_{hyper}) \cdot f_{CNG}$$

This is an instantaneous algebraic relation: when cGMP = 1 (dark), V = V_dark; when cGMP = 0 (saturated light), V = V_hyper.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -40.0 | mV | Membrane potential |
| `v_dark` | -40.0 | mV | Dark resting potential |
| `v_hyper` | -70.0 | mV | Maximum hyperpolarised potential |
| `cgmp` | 1.0 | — | Normalised cGMP concentration [0, 1] |
| `tau_act` | 20.0 | ms | Activation (hydrolysis) time constant |
| `tau_rec` | 500.0 | ms | Recovery (resynthesis) time constant |
| `sensitivity` | 0.01 | — | Light-to-PDE coupling gain |
| `dt` | 0.1 | ms | Integration timestep |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/sensory.rs` |
| PyO3 wrapper | `py_sensory_graded!` macro in `pyo3_neurons.rs` |
| NetworkRunner wired | **No** — graded model, `step()` returns `f64` |
| `create_neuron("RodPhotoreceptor")` | No (not in NetworkRunner variant enum) |
| STRONG tests | 5 (hyperpolarise, dark stability, slow recovery, cGMP bounded, performance) |
| NaN/extreme input test | Bounded test covers 10k steps at extreme input; negative light clamped to 0 |
| Benchmark | `rod_10k_steps`: **308 µs** (30.8 ns/step), i5-11600K |

Graded sensory models are accessed directly via PyO3 (`RodPhotoreceptor` class) or Rust. They are not routed through `NetworkRunner` because the network step loop expects `i32` spike outputs.

---

## Benchmark (i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| rod_10k_steps | 308 µs |
| Per step | **30.8 ns** |

The performance test asserts 100k steps complete in <50 ms. The step function evaluates one `powi(3)`, one `clamp`, and simple arithmetic. Expected cost under 500 ns/step.

---

## Findings

1. **tau_rec >> tau_act produces asymmetric kinetics.** Activation (cGMP hydrolysis) is fast (20 ms), but recovery (resynthesis) is very slow (500 ms). This matches rod dark adaptation timescales.
2. **Hill coefficient n=3 produces sharp transition.** The cubic CNG dependence on cGMP means small cGMP changes near threshold produce large current changes.
3. **Light saturation is robust.** At high light intensities, cGMP clamps to 0 and V reaches V_hyper = -70 mV. No numerical instability.
4. **Negative light inputs are clamped to zero.** The model handles invalid inputs gracefully.
