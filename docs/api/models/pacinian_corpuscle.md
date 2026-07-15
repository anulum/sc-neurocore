<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — PacinianCorpuscle model reference -->
# PacinianCorpuscle

**Module:** `engine/src/neurons/sensory/pacinian_corpuscle.rs`
**Reference:** Loewenstein & Skalak 1966; Bell et al. 1994
**Family:** Spiking sensory receptor, rapidly adapting type II (RAII) mechanoreceptor
**State variables:** `v` (membrane potential), `prev_pressure` (previous input), `adapt` (fast adaptation variable)

---

## Biological Context

Pacinian corpuscles are rapidly adapting (RA/RAII) mechanoreceptors located deep in the dermis and subcutaneous tissue, as well as in joint capsules and periosteum. Their lamellated structure acts as a mechanical high-pass filter.

Key features:
- Respond to vibration and transient pressure changes, not sustained pressure
- Lamellar capsule filters out static components: only dynamic (onset/offset) signals reach the nerve terminal
- Best frequency response at 200-300 Hz (Meissner corpuscles handle lower frequencies)
- Derivative-like response: firing rate encodes rate of change of pressure, not absolute pressure
- Very rapid adaptation: stops firing within milliseconds of pressure becoming constant
- Large receptive fields (~10-20 mm), deep location

The model computes the time derivative of pressure input, applies gain, and drives a LIF neuron with fast adaptation. The `abs()` on the derivative means both onset and offset of pressure produce excitation.

---

## Equations

### Pressure derivative (high-pass filtering)

$$\frac{dP}{dt} \approx \frac{P(t) - P(t - dt)}{dt}$$

### Drive current

$$drive = gain \cdot \left|\frac{dP}{dt}\right| - w$$

### Membrane dynamics

$$\frac{dV}{dt} = \frac{-(V - V_{rest}) + drive}{\tau}$$

The maintained Rust engine uses exact first-order relaxation over each
timestep while holding the derivative drive fixed:

$$V_\infty = V_{rest} + drive$$

$$V(t+dt) = V_\infty + (V(t)-V_\infty)\exp(-dt/\tau)$$

### Fast adaptation

$$\frac{dw}{dt} = \frac{0.5 \cdot \max(drive, 0) - w}{\tau_{adapt}}$$

$$w_\infty = 0.5 \cdot \max(drive, 0)$$

$$w(t+dt) = w_\infty + (w(t)-w_\infty)\exp(-dt/\tau_{adapt})$$

### Spike and reset

$$\text{if } V \geq V_{threshold}: \quad V \leftarrow V_{reset}, \quad \text{emit spike (1)}$$

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `v_rest` | -65.0 | mV | Resting potential |
| `v_reset` | -70.0 | mV | Post-spike reset potential |
| `v_threshold` | -50.0 | mV | Spike threshold |
| `tau` | 2.0 | ms | Membrane time constant |
| `prev_pressure` | 0.0 | — | Previous pressure input (for derivative) |
| `adapt` | 0.0 | — | Fast adaptation variable |
| `tau_adapt` | 5.0 | ms | Adaptation time constant |
| `gain` | 10.0 | — | Derivative-to-current gain |
| `dt` | 0.5 | ms | Integration timestep |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/sensory/pacinian_corpuscle.rs` |
| PyO3 wrapper | `py_neuron_default!` macro in `pyo3_neurons.rs` |
| NetworkRunner wired | `NeuronVariant::Pacinian` |
| `create_neuron("Pacinian")` or `create_neuron("PacinianCorpuscle")` | Yes |
| coverage tests | 10 (onset firing, adaptation, rest, reset, exact relaxation, invalid input/state/voltage, non-finite candidate, constructor/default equivalence) |
| NaN/extreme input test | Module-owned Pacinian tests plus NetworkRunner `all_models_*` tests |
| Benchmark | `pacinian_10k_steps`: **489.26 µs** (48.93 ns/step), i5-11600K |

---

## Benchmark (i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| pacinian_10k_steps | 489.26 µs |
| Per step | **48.93 ns** |

The step function evaluates the pressure derivative, `abs()`, two exact
first-order relaxations, state validation, and one comparison. Artifact:
`benchmarks/results/local_i5_11600k_criterion_2026-05-31_pacinian_corpuscle.json`.

---

## Findings

1. **Derivative-like response confirmed.** The `pacinian_fires_on_pressure_onset` test shows spikes during a pressure ramp, while `pacinian_adapts_to_sustained` shows cessation during constant pressure (dP/dt = 0).
2. **abs(dP/dt) makes response symmetric.** Both pressure onset and offset produce excitation, matching biological observations of ON/OFF responses in Pacinian afferents.
3. **Fast adaptation (tau_adapt = 5 ms) complements the derivative.** Even if dP/dt is sustained (e.g. linear ramp), the adaptation variable suppresses prolonged firing through exact first-order relaxation.
4. **High gain (10.0) compensates for derivative scaling.** The dP/dt value can be small for gradual changes; the high gain ensures suprathreshold drive for physiologically relevant rates of change.
5. **Reset clears prev_pressure and adapt.** This prevents artefactual dP/dt spikes when re-starting after a stimulus history.
6. **Fail-closed boundary.** Non-finite pressure, non-finite state, nonphysical finite voltage, invalid time constants, negative adaptation/gain parameters, and invalid threshold ordering return no spike without mutation.
