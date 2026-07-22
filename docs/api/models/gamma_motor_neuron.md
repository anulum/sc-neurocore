<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore - GammaMotorNeuron model documentation -->

# GammaMotorNeuron

**Module:** `engine/src/neurons/motor/gamma_motor_neuron.rs`
**Reference:** Prochazka & Hulliger, Prog. Brain Res. 80, 1989; Taylor et al., J. Physiol. 519(3), 1999
**Family:** Leaky integrate-and-fire with adaptation, fusimotor neuron
**State variables:** `v` (membrane potential), `adapt` (slow adaptation current)

---

## Biological Context

Gamma motor neurons innervate intrafusal muscle fibres within muscle spindles, regulating proprioceptive sensitivity without producing extrafusal force. They form the efferent arm of the fusimotor system, adjusting spindle tension to maintain sensitivity across the full range of muscle lengths and velocities.

Two functional subtypes:
- **Dynamic gamma** (bag1 fibres): enhances velocity sensitivity of primary (Ia) afferents. Higher firing rate, weaker adaptation.
- **Static gamma** (bag2/chain fibres): enhances length sensitivity of both primary (Ia) and secondary (II) afferents. Lower firing rate, stronger adaptation.

Key electrophysiological features:
- Smaller soma than alpha motor neurons → lower rheobase
- Lower sustained firing rates (5-30 Hz) than alpha (10-50 Hz)
- Spike-frequency adaptation via slow K+ current
- No persistent inward current (unlike alpha)
- Tonic firing pattern (no bursting)

---

## Equations

### Membrane potential

$$\tau \frac{dV}{dt} = -(V - V_{rest}) + g \cdot \max(drive, 0) - adapt$$

### Spike and reset

$$V \geq V_{threshold}: \quad V \leftarrow V_{reset}$$

### Adaptation (slow K+ analogue)

$$\tau_{adapt} \frac{d(adapt)}{dt} = a_{adapt} (V - V_{rest}) - adapt$$

The maintained implementation applies the closed-form relaxation for each
linear first-order state over one timestep, preserving the continuous-time
leak/adaptation contract while rejecting invalid parameters, non-finite drive,
and non-finite candidate states before mutation.

---

## Parameters

| Parameter | Dynamic | Static | Unit | Description |
|-----------|---------|--------|------|-------------|
| `v` | -65.0 | -65.0 | mV | Membrane potential |
| `v_rest` | -65.0 | -65.0 | mV | Resting potential |
| `v_reset` | -70.0 | -70.0 | mV | Post-spike reset |
| `v_threshold` | -50.0 | -50.0 | mV | Spike threshold |
| `tau` | 8.0 | 12.0 | ms | Membrane time constant |
| `adapt` | 0.0 | 0.0 | mV | Adaptation current |
| `tau_adapt` | 100.0 | 200.0 | ms | Adaptation time constant |
| `a_adapt` | 0.3 | 0.5 | — | Adaptation coupling |
| `gain` | 1.0 | 1.0 | — | Input scaling |
| `dynamic` | true | false | — | Subtype flag |
| `dt` | 0.5 | 0.5 | ms | Integration timestep |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/motor/gamma_motor_neuron.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` |
| NetworkRunner wired | `NeuronVariant::GammaMotor` |
| `create_neuron("GammaMotor")` | Yes (creates dynamic subtype) |
| `supported_models()` | Includes "GammaMotor" |
| coverage tests | Module-specific Python tests plus Rust inline and Go service tests for continuous relaxation, subtype firing, invalid-parameter rejection, non-finite drive preservation, and corrupted-state preservation |
| Pipeline integration | Covered by `create_neuron_all_supported` |
| Benchmark | `gamma_motor_10k_steps`: **1.21 ms** (121 ns/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| gamma_motor_10k_steps | 1.21 ms |
| Per step | **121 ns** |

Simple LIF with adaptation and closed-form first-order relaxation. Historical
timing was measured 2026-04-04; regenerate before using as release evidence.

---

## Comparison with Related Models

| Property | Gamma (this) | Alpha Motor | MerkelCell | LIF |
|----------|-------------|-------------|------------|-----|
| PIC | No | Yes | No | No |
| AHP (Ca2+) | No | Yes | No | No |
| Adaptation | Yes (slow K+) | Yes (SK) | Yes (slow) | No |
| Sub-stepping | No | 50 | No | No |
| Subtypes | Dynamic/Static | — | — | — |
| Soma size | Small | Large | — | — |

---

## Findings

1. **Dynamic fires more than static.** With identical input (20.0), dynamic subtype produces more spikes due to weaker adaptation (tau_adapt=100 vs 200, a_adapt=0.3 vs 0.5). Verified.
2. **Adaptation reduces rate.** Later epochs produce no more spikes than earlier epochs at constant input. Verified.
3. **Negative input clamped.** `drive.max(0.0)` prevents negative fusimotor drive from affecting the neuron.
4. **Reset deterministic.** Post-reset neuron matches fresh neuron exactly.
5. **Fail-closed runtime boundary.** Non-finite drive or corrupted runtime state preserves the previous state and reports no spike on Rust/Go safety paths; Python raises before mutation.
