<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (C) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore MotorUnit model reference -->

# MotorUnit

**Module:** `engine/src/neurons/motor.rs`
**Reference:** Fuglevand et al., J. Neurophysiol. 70(6), 1993; Heckman & Enoka, Compr. Physiol. 2(4), 2012
**Family:** LIF motor neuron with exact adaptation relaxation and muscle fibre force model
**State variables:** `v` (membrane potential), `adapt` (adaptation current), `force` (normalised force output)

---

## Biological Context

A motor unit is the functional unit of motor control: one alpha motor neuron and all the muscle fibres it innervates. Each spike triggers a twitch contraction. At low firing rates, individual twitches are visible. At higher rates, twitches fuse into smooth force (tetanus). This rate coding mechanism is the primary way the nervous system controls force output.

Two subtypes modelled:
- **Slow (type S):** small motor neuron, low force twitches, fatigue-resistant. Recruited first (Henneman size principle).
- **Fast (type FF):** large motor neuron, high force twitches, fatigable. Recruited last for maximal effort.

Key features:
- Spike → muscle twitch (additive force increment)
- Force decays exponentially between spikes
- Force saturates at 1.0 (normalised)
- Rate coding: higher firing rate → more twitch overlap → higher force
- Adaptation limits sustained firing rate

---

## Equations

### Motor neuron (LIF with adaptation)

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + g \cdot \max(drive, 0) - adapt$$

$$\tau_{adapt} \frac{d(adapt)}{dt} = a_{adapt}(V - V_{rest}) - adapt$$

### Exact numerical update

The runtime uses the closed-form LIF membrane relaxation for fixed drive and adaptation, then updates adaptation by closed-form first-order relaxation toward (V - V_{rest})1 Candidate states are committed transactionally only if voltage, force, and parameters remain finite and physiological. Invalid, infinite, or excess drive returns no spike and preserves the pre-step state.

### Muscle force model

On each spike:

$$force \leftarrow \min(force + A_{twitch}, 1.0)$$

Between spikes:

$$force \leftarrow force \cdot \exp(-dt / \tau_{twitch})$$

where $A_{twitch}$ is the peak twitch amplitude and $\tau_{twitch}$ is the contraction time.

---

## Parameters

| Parameter | Slow (S) | Fast (FF) | Unit | Description |
|-----------|----------|-----------|------|-------------|
| `v` | -65.0 | -65.0 | mV | Membrane potential |
| `v_rest` | -65.0 | -65.0 | mV | Resting potential |
| `v_reset` | -70.0 | -70.0 | mV | Post-spike reset |
| `v_threshold` | -50.0 | -50.0 | mV | Spike threshold |
| `tau_m` | 10.0 | 6.0 | ms | Membrane time constant |
| `adapt` | 0.0 | 0.0 | mV | Adaptation variable |
| `tau_adapt` | 100.0 | 50.0 | ms | Adaptation time constant |
| `a_adapt` | 0.2 | 0.1 | — | Adaptation coupling |
| `gain` | 1.0 | 1.0 | — | Input scaling |
| `force` | 0.0 | 0.0 | — | Normalised force [0, 1] |
| `twitch_amp` | 0.05 | 0.3 | — | Peak twitch amplitude |
| `tau_twitch` | 90.0 | 30.0 | ms | Contraction time |
| `dt` | 0.5 | 0.5 | ms | Integration timestep |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/motor.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, adapt, force) |
| NetworkRunner wired | `NeuronVariant::MotorUnitCell` |
| `create_neuron("MotorUnit")` | Yes (creates slow subtype) |
| `supported_models()` | Includes "MotorUnit" |
| Behavior tests | Rust engine 15; Python model 4; Go service 4; Rust safety 6 |
| Pipeline integration | Covered by `create_neuron_all_supported` |
| Benchmark | `motor_unit_10k_steps`: **327 µs** median (32.7 ns/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| motor_unit_10k_steps | 327 µs |
| Per step | **32.7 ns** |

Exact LIF membrane relaxation, exact adaptation relaxation, and exact exponential force decay; no sub-stepping. Measured 2026-05-31 on local i5-11600K.

---

## Findings

1. **Force increases during spiking.** force > 0 after sustained input. Verified.
2. **Force decays without input.** force decreases when spikes stop. Verified.
3. **Fast MU produces more force.** twitch_amp=0.3 vs 0.05. Verified.
4. **Force capped at 1.0.** Even with prolonged maximal drive. Verified.
5. **Rate coding verified.** Higher input → higher firing rate → more force.
6. **Invalid-drive fail-closed behavior.** NaN, infinite, excess-drive, and invalid-parameter paths preserve pre-step state and return no spike. Verified.
