<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# WilsonHRNeuron

**Module:** `sc_neurocore.neurons.models.wilson_hr`
**Rust engine:** `sc_neurocore_engine::neurons::simple_spiking::WilsonHRNeuron`
**Reference:** Wilson, H. R. (1999), *Spikes, Decisions, and Actions*, Oxford University Press.

`WilsonHRNeuron` implements Wilson's two-state polynomial cortical model. The
model keeps voltage `v` and recovery `r` dimensionless and uses polynomial
coefficients from the Wilson-HR formulation rather than explicit ionic gates.

## State equation

```text
dV/dt = -(17.81 + 47.71*V + 32.63*V^2)*(V - 0.55) - 26*R*(V + 0.92) + I
dR/dt = (-R + 1.35*V + 1.03) / tau_R
```

Default parameters:

| Parameter | Default | Contract |
|-----------|---------|----------|
| `v` | `-0.7` | finite |
| `r` | `0.1` | finite |
| `tau_r` | `1.9` | finite, positive |
| `v_peak` | `0.4` | finite |
| `dt` | `0.05` | finite, positive |

A spike is reported when the RK4 candidate voltage reaches `v_peak`. The model
then applies the Wilson-HR hard voltage reset `v <- -0.7` while preserving the
candidate recovery value.

## Integration contract

The maintained Python, Rust engine, Go service, Julia service, and Rust safety
surfaces use candidate-first RK4 over the coupled `(v, r)` state. Each stage
evaluates the same Wilson-HR polynomial right-hand side. State is committed only
after both candidate variables are finite.

Invalid runtime contracts are fail-closed:

| Condition | Python public surface | Go / Julia / Rust safety / Rust engine |
|-----------|-----------------------|----------------------------------------|
| Non-finite state | raises before mutation | returns sentinel / preserves state |
| Non-finite current | raises before mutation | returns sentinel / preserves state |
| Invalid `tau_r` or `dt` | raises before mutation | returns sentinel / preserves state |
| Non-finite polynomial, derivative, or candidate | raises before mutation | returns sentinel / preserves state |

## Public workflow contract

| Surface | Contract |
|---------|----------|
| `tests/test_model_wilson_hr.py` | Python RK4 reference, polynomial equation, dynamics, validation, public network workflow |
| `engine/src/neurons/simple_spiking.rs` | Rust engine RK4 candidate and invalid-state preservation |
| `src/sc_neurocore/accel/go/services/wilson_hr.go` | Go RK4 service and invalid-state preservation |
| `src/sc_neurocore/accel/julia/neurons/wilson_hr.jl` | Julia RK4 service and invalid-state sentinel behavior |
| `src/sc_neurocore/accel/rust/safety/wilson_hr.rs` | Standalone Rust safety RK4 parity and invalid-state preservation |

The cross-module Python workflow is explicitly named as the Wilson-HR public
surface inside the Python simulator. It exercises real `Population`,
`Projection`, `Network`, `SpikeMonitor`, and spike-stat analysis APIs.

## Example

```python
from sc_neurocore.neurons.models.wilson_hr import WilsonHRNeuron

neuron = WilsonHRNeuron()
spikes = []
for step in range(50_000):
    if neuron.step(0.3):
        spikes.append(step)
```

## Benchmark evidence

| Benchmark | Steps | Median | Per step | Artifact |
|-----------|-------|--------|----------|----------|
| `wilson_hr_10k_steps` | 10,000 | 514.29 us | 51.4 ns | `benchmarks/results/local_i5_11600k_criterion_2026-05-31_wilson_hr.json` |

| Benchmark | Steps | Median per step | Mean per step | Artifact |
|-----------|-------|-----------------|---------------|----------|
| `python_wilson_hr_rk4_reference` | 100,000 | 8.197 us | 8.297 us | `benchmarks/results/local_i5_11600k_python_2026-05-31_wilson_hr.json` |

Benchmarks are regression evidence only. The acceptance contract is the
Wilson-HR equation, RK4 candidate parity, finite-state validation, and public
workflow behavior.

## Scientific scope

Wilson-HR is appropriate when a two-state polynomial cortical spiking model is
needed without explicit Hodgkin-Huxley gating variables. It is not a detailed
ion-channel model; it is a validated polynomial dynamical system surface with
candidate-first integration and cross-runtime parity checks.
