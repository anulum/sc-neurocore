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
| `tests/test_wilson_hr_backends.py` | Cross-backend `simulate` parity (rust/julia/go bit-exact, mojo ULP-bounded) |
| `engine/src/neurons/simple_spiking.rs` | Rust engine RK4 `step`/`simulate` candidate and invalid-state preservation |
| `src/sc_neurocore/accel/julia/neurons/wilson_hr.jl` | Julia RK4 `simulate_trace` parity |
| `src/sc_neurocore/accel/go/neurons/wilson_hr/wilson_hr.go` | Go RK4 c-shared `simulate` parity |
| `src/sc_neurocore/accel/mojo/neurons/wilson_hr.mojo` | Mojo RK4 FFI `simulate` (FMA ULP-bounded) |
| `src/sc_neurocore/accel/rust/safety/wilson_hr.rs` | Standalone Rust safety RK4 parity and invalid-state preservation |

The cross-module Python workflow is explicitly named as the Wilson-HR public
surface inside the Python simulator. It exercises real `Population`,
`Projection`, `Network`, `SpikeMonitor`, and spike-stat analysis APIs.

## Polyglot acceleration

A single `step` is trivial, but an N-step run is a sequential RK4 recurrence that
does not vectorise, so a compiled inner loop genuinely beats Python.
`simulate(n_steps, current, backend="auto")` dispatches across the polyglot chain
and returns `(trace, spikes)` (the `v` trace is already hard-reset to `-0.7` on
spiking steps):

```python
from sc_neurocore.neurons.models.wilson_hr import WilsonHRNeuron

neuron = WilsonHRNeuron()
trace, spikes = neuron.simulate(50_000, current=10.0)   # auto → Rust
```

The right-hand side is exact polynomial floating-point arithmetic (no
transcendental functions), so **Rust, Julia and Go reproduce the NumPy reference
bit-for-bit**. Mojo's release build contracts the RK4 multiply-adds into fused
multiply-adds; the per-spike hard reset re-anchors the trajectory and a
two-dimensional autonomous flow cannot be chaotic (Poincaré-Bendixson), so the
single-ULP difference does not accumulate — the whole-trace gap stays near one ULP
even over millions of steps and the spike counts match. `auto` selects Rust (the
fastest bit-exact backend, shipped in the wheel).

### Measured throughput

2,000,000 RK4 steps, suprathreshold regime (`current=10.0`), median of 5 repeats.
Non-isolated loaded workstation (Intel i5-11600K) per
`BROADCAST_2026-06-04_benchmark_core_isolation` — functional/regression evidence,
not an isolated-core figure. Reproduce with
`python benchmarks/bench_wilson_hr_simulate.py`.

| Backend | Median (ms) | Speed-up vs Python | Whole-trace parity |
|---------|------------:|-------------------:|--------------------|
| python  | 4160.43 | 1.0× | reference |
| mojo    | 82.82 | 50.2× | 2.8×10⁻¹⁶ (FMA, ~1 ULP) |
| go      | 116.65 | 35.7× | bit-exact (0) |
| rust (`auto`) | 129.89 | 32.0× | bit-exact (0) |
| julia   | 138.63 | 30.0× | bit-exact (0) |

Artefact: `benchmarks/results/bench_wilson_hr_simulate.json`.

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
