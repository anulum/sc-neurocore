# SCIzhikevichNeuron

**Module:** `sc_neurocore.neurons.sc_izhikevich`
**Class:** `SCIzhikevichNeuron`
**Reference:** Izhikevich, E. M. (2003), *IEEE Transactions on Neural Networks* 14(6), 1569-1572
**Family:** Two-variable quadratic integrate-and-fire model
**State variables:** `v` (membrane potential), `u` (recovery variable)
**Maintained accelerated path:** RK4 parity harness through Python, Rust, Julia, Go, and Mojo wrappers

This page documents the maintained software neuron, its validation contract,
the explicit integrator choices, and the latest measured RK4 parity benchmark.
The historical half-Euler path remains the default for compatibility; the RK4
path is the cross-language reference path used by the benchmark harness.

---

## Scientific Model

Izhikevich's 2003 model uses a quadratic membrane equation and a linear
recovery variable:

$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I$$

$$\frac{du}{dt} = a(bv - u)$$

When the membrane reaches the spike threshold, the model emits a spike and
applies the reset map:

$$v \geq 30 \Rightarrow v \leftarrow c,\quad u \leftarrow u + d$$

The implementation follows that formulation directly. Additive stochasticity
is optional and is applied after deterministic integration but before threshold
and reset handling. With `noise_std=0.0`, the neuron is deterministic.

## Parameters

| Parameter | Default | Unit | Description |
|---|---:|---|---|
| `a` | 0.02 | dimensionless | Recovery time scale |
| `b` | 0.2 | dimensionless | Sensitivity of `u` to subthreshold `v` |
| `c` | -65.0 | mV-style model unit | Post-spike membrane reset |
| `d` | 8.0 | model current unit | Post-spike recovery increment |
| `dt` | 1.0 | model time step | Integration step used by both maintained paths |
| `noise_std` | 0.0 | membrane unit | Standard deviation for additive voltage noise |
| `seed` | `None` | — | Seed for deterministic local RNG when noise is enabled |
| `integrator` | `baseline_half_euler` | — | `baseline_half_euler` or `rk4` |

Default `a`, `b`, `c`, and `d` are the regular-spiking parameter set used by
the project constants. `reset_state()` initialises `v = c` and `u = b * c`.

## Integrator Paths

`SCIzhikevichNeuron` exposes two explicit integration paths:

| Path | Constructor value | Role | Compatibility |
|---|---|---|---|
| Historical default | `baseline_half_euler` | Two half-Euler updates per public `step()` call | Preserves previous public behaviour |
| Cross-language reference | `rk4` | Explicit fourth-order Runge-Kutta integration | Used for parity and timing evidence |

Example:

```python
from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron

baseline = SCIzhikevichNeuron(noise_std=0.0)
rk4 = SCIzhikevichNeuron(noise_std=0.0, integrator="rk4")

baseline_spike = baseline.step(10.0)
rk4_spike = rk4.step(10.0)
```

The default is intentionally not changed by the RK4 work. Compatibility tests
assert that a default instance follows the historical half-Euler path.

## Input Contract

The maintained Python path rejects invalid numerical input before the neuron
state can be poisoned:

| Input | Accepted values | Rejection |
|---|---|---|
| `a`, `b`, `c`, `d` | finite `int` or `float` scalar | `ValueError` |
| `dt` | positive finite scalar | `ValueError` |
| `noise_std` | non-negative finite scalar | `ValueError` |
| `integrator` | `baseline_half_euler` or `rk4` | `ValueError` |
| `step(input_current)` | finite scalar current | `ValueError` |

The maintained RK4 parity wrappers for Rust, Julia, Go, and Mojo also reject
empty traces, non-1D traces, non-finite current traces, and invalid `dt` before
calling their backend kernels.

Legacy generated mirror files are not the authoritative contract unless routed
through the maintained wrappers documented here and in
[Neuron Integrator Paths](../../guides/neuron_integrator_paths.md).

## Behavioural Expectations

### Regular spiking

With regular-spiking defaults and constant current, the model produces repeated
spikes. The focused integrator tests require both maintained paths to spike
under a 1,000-step constant-current probe and constrain the RK4 spike count
against the historical path.

### Reset map

Every spike applies the Izhikevich reset map exactly:

- `v` is set to `c`
- `u` is incremented by `d`
- `step(...)` returns `1`

If the membrane remains below threshold, `step(...)` returns `0`.

### Noise

When `noise_std > 0`, stochastic voltage perturbation is applied on both
integrator paths. Tests cover noise reachability with a fixed seed so that
the random path is not silently disconnected.

### State reporting

`get_state()` returns a dictionary containing the current `v` and `u` as
floating-point values. The public state shape is stable across both integrator
paths.

## Python Implementation Shape

The deterministic right-hand side is shared by both integrator paths:

```python
def _rhs(self, v: float, u: float, input_current: float) -> tuple[float, float]:
    dv = 0.04 * v**2 + 5.0 * v + 140.0 - u + input_current
    du = self.a * (self.b * v - u)
    return dv, du
```

The historical path splits each public step into two half steps:

```python
half_dt = self.dt * 0.5
for _ in range(2):
    dv, du = self._rhs(self.v, self.u, input_current)
    self.v += dv * half_dt
    self.u += du * half_dt
```

The RK4 path integrates the two-dimensional state vector with the same
right-hand side and then applies the same noise, threshold, and reset handler.

## Cross-Language RK4 Contract

The RK4 benchmark case for this neuron uses:

| Field | Value |
|---|---:|
| Model key | `izhikevich` |
| Display name | `Izhikevich` |
| State keys | `v`, `u` |
| Parity `dt` | 1.0 |
| Parity trace length | 1,000 steps |
| Timing trace length | 10,000 steps |
| Timing repeats | 3 |
| Parity tolerance | 1e-9 |
| Current trace | constant `1.0` |

The benchmark compares backend state traces and spike indices against the
Python RK4 reference. A backend is counted as usable only when it is available,
executes the case, preserves spike indices, and stays within the configured
state tolerance.

## Latest Benchmark Evidence

Latest local benchmark artifact:

- `benchmarks/results/bench_neuron_integrators_2026-05-09T1848.json`
- `benchmarks/results/bench_neuron_integrators.json`

Run metadata:

| Field | Value |
|---|---|
| Timestamp UTC | 2026-05-09T16:49:04Z |
| CPU | 11th Gen Intel(R) Core(TM) i5-11600K @ 3.90GHz |
| Platform | Linux-6.17.0-23-generic-x86_64-with-glibc2.39 |
| Python | 3.12.3 |
| Parity steps | 1,000 |
| Timing steps | 10,000 |
| Repeats | 3 |

### Izhikevich RK4 timing

| Backend | Used | Best wall time | Steps/s | Speedup vs Python |
|---|---|---:|---:|---:|
| Python | yes | 92.704 ms | 107,870 | 1.000x |
| Rust | yes | 0.295 ms | 33,862,950 | 314.251x |
| Julia | yes | 0.306 ms | 32,629,835 | 302.954x |
| Go | yes | 0.316 ms | 31,603,865 | 293.367x |
| Mojo | yes | 0.283 ms | 35,316,098 | 327.576x |

### Izhikevich RK4 parity

| Backend | Within tolerance | Spike indices equal | Max absolute state delta | Bit-exact |
|---|---|---|---:|---|
| Python | yes | yes | 0.0 | yes |
| Rust | yes | yes | 0.0 | yes |
| Julia | yes | yes | 0.0 | yes |
| Go | yes | yes | 5.684341886080802e-14 | no |
| Mojo | yes | yes | 5.684341886080802e-14 | no |

The Go and Mojo deltas are well below the 1e-9 tolerance and reflect ordinary
floating-point expression-order differences, not a behavioural divergence.

## Test Coverage

Focused tests for this neuron and its maintained RK4 path are:

| Test | Purpose |
|---|---|
| `test_izhikevich_integrator_validation` | rejects unsupported integrator names |
| `test_izhikevich_rejects_invalid_numerical_configuration` | rejects non-finite or invalid constructor numerics |
| `test_izhikevich_rejects_non_finite_input_current` | rejects invalid runtime current on both paths |
| `test_izhikevich_rk4_regular_spiking_and_default_preserved` | checks regular spiking, RK4 tracking, and unchanged default path |
| `test_izhikevich_noise_injection_is_reachable_on_both_paths` | ensures noise path is wired on both integrators |
| `test_izhikevich_get_state_reflects_running_v_and_u` | checks public state shape and values |
| `test_rust_rk4_neuron_simulator_rejects_invalid_trace_or_dt` | Rust wrapper fail-closed trace and `dt` validation |
| `test_go_rk4_neuron_simulator_rejects_invalid_trace_or_dt` | Go wrapper fail-closed trace and `dt` validation |
| `test_julia_rk4_neuron_simulator_rejects_invalid_trace_or_dt` | Julia wrapper fail-closed trace and `dt` validation |
| `test_mojo_rk4_neuron_simulator_rejects_invalid_trace_or_dt` | Mojo wrapper fail-closed trace and `dt` validation |

The broader Izhikevich pattern suite remains relevant for the historical
firing-pattern contract:

- `tests/test_izhikevich_20_patterns.py`
- `docs/validation/izhikevich_patterns.md`

## Verification Commands

Scoped verification used for the latest update:

```bash
python -m pytest tests/test_izhikevich_integrator_paths.py
python -m pytest tests/test_rust_rk4_neuron_parity.py
python -m pytest tests/test_go_rk4_neuron_parity.py
python -m pytest tests/test_julia_rk4_neuron_parity.py
python -m pytest tests/test_mojo_rk4_neuron_parity.py
```

Wheel and Rust-engine verification:

```bash
python -m maturin build --manifest-path engine/Cargo.toml --release
cargo test -p sc_neurocore_engine rk4_neurons --quiet
```

Benchmark command used to generate the artifact:

```bash
python benchmarks/bench_neuron_integrators.py --json benchmarks/results/bench_neuron_integrators_2026-05-09T1848.json
```

Documentation verification:

```bash
PYTHONPATH=src python -m mkdocs build --strict
```

## Firing-Pattern Validation

The historical firing-pattern contract is validated separately from the RK4
backend benchmark. It checks whether parameter regimes derived from Izhikevich
2003 and Izhikevich 2004 produce the expected qualitative spike classes over
500 ms simulations with `dt=0.5` and `noise_std=0.0`.

### Test parameter regimes

| Pattern | `a` | `b` | `c` | `d` | Input | Minimum expected spikes |
|---|---:|---:|---:|---:|---:|---:|
| RS | 0.02 | 0.2 | -65 | 8 | 10 | 5 |
| IB | 0.02 | 0.2 | -55 | 4 | 10 | 5 |
| CH | 0.02 | 0.2 | -50 | 2 | 10 | 10 |
| FS | 0.1 | 0.2 | -65 | 2 | 10 | 20 |
| TC tonic | 0.02 | 0.25 | -65 | 0.05 | 5 | 1 |
| LTS | 0.02 | 0.25 | -65 | 2 | 10 | 10 |
| Tonic spiking | 0.02 | 0.2 | -65 | 6 | 14 | 5 |
| Tonic bursting | 0.02 | 0.2 | -50 | 2 | 15 | 10 |
| Mixed mode | 0.02 | 0.2 | -55 | 4 | 10 | 5 |
| Spike-frequency adaptation | 0.01 | 0.2 | -65 | 8 | 30 | 5 |
| Class 1 excitability | 0.02 | -0.1 | -55 | 6 | 30 | 5 |
| Class 2 excitability | 0.2 | 0.26 | -65 | 0 | 5 | 10 |
| Spike latency | 0.02 | 0.2 | -65 | 6 | 7 | 1 |
| Accommodation | 0.02 | 1.0 | -55 | 4 | 10 | 10 |
| Inhibition-induced spiking | -0.02 | -1.0 | -60 | 8 | 80 | 1 |

Silent or near-silent regimes at zero current are checked separately:

| Pattern | `a` | `b` | `c` | `d` | Input | Expected result |
|---|---:|---:|---:|---:|---:|---|
| TC burst | 0.02 | 0.25 | -65 | 0.05 | 0 | at most two spikes |
| RZ | 0.1 | 0.26 | -65 | 2 | 0 | at most two spikes |

### Measured legacy validation record

The older validation page records a 2026-03-28 run with these measured values:

| Pattern | Input | Spikes / 500 ms | ISI | CV |
|---|---:|---:|---:|---:|
| RS | 10 | 12 | 45.0 ms | 0.123 |
| IB | 10 | 16 | 31.2 ms | — |
| CH | 10 | 37 | 13.1 ms | — |
| FS | 10 | 50 | 10.0 ms | — |
| TC burst | 0 | 0 | — | — |
| TC tonic | 5 | 46 | 10.8 ms | — |
| RZ | 0 | 1 | — | — |
| LTS | 10 | 35 | 14.5 ms | — |
| Tonic spiking | 14 | 19 | 26.8 ms | — |
| Phasic spiking | 0.5 | 1 | — | — |
| Tonic bursting | 15 | 58 | 8.5 ms | — |
| Phasic bursting | 0.6 | 1 | — | — |
| Mixed mode | 10 | 16 | 31.2 ms | — |
| Spike-frequency adaptation | 30 | 20 | 24.9 ms | — |
| Class 1 | 30 | 15 | 33.1 ms | — |
| Class 2 | 5 | 53 | 9.5 ms | — |
| Spike latency | 7 | 9 | 55.6 ms | — |
| Subthreshold oscillation | 2 | 25 | 20.3 ms | — |
| Accommodation | 10 | 84 | 6.0 ms | — |
| Inhibition-induced | 80 | 1 | — | — |

This older pattern evidence is not a replacement for the 2026-05-09 RK4
benchmark. It validates qualitative firing regimes; the newer benchmark
validates cross-language RK4 numerical parity and speed.

## Pipeline Wiring

The maintained pipeline touchpoints for this neuron are:

| Layer | Path | Role |
|---|---|---|
| Public Python neuron | `src/sc_neurocore/neurons/sc_izhikevich.py` | primary user-facing class |
| Pattern validation | `tests/test_izhikevich_20_patterns.py` | qualitative model-regime checks |
| Integrator validation | `tests/test_izhikevich_integrator_paths.py` | baseline/RK4 contract and guards |
| Rust RK4 bridge | `engine/src/rk4_neurons.rs` | PyO3-backed RK4 parity simulator |
| Go RK4 wrapper | `src/sc_neurocore/accel/go/rk4_neurons/__init__.py` | maintained Go parity entry point |
| Julia RK4 wrapper | `src/sc_neurocore/accel/julia/neurons/__init__.py` | maintained Julia parity entry point |
| Mojo RK4 wrapper | `src/sc_neurocore/accel/mojo/rk4_neurons/__init__.py` | maintained Mojo parity entry point |
| Benchmark harness | `benchmarks/bench_neuron_integrators.py` | cross-language parity and timing |
| User guide | `docs/guides/neuron_integrator_paths.md` | integrator-wide explanation |
| Audit evidence | `docs/internal/AUDIT_INDEX.md` | internal bounded F4 evidence |

The user-facing Python API is the root contract. Backend wrappers must match
the Python RK4 reference and must fail closed on invalid traces instead of
letting NaN, infinity, empty arrays, or malformed ranks reach native kernels.

## Failure Modes and Guards

| Failure mode | Guard location | Expected behaviour |
|---|---|---|
| Unsupported integrator string | Python constructor | `ValueError` before state initialisation completes |
| Non-finite `a`, `b`, `c`, or `d` | Python constructor | `ValueError` naming the invalid parameter |
| Zero or negative `dt` | Python constructor and backend wrappers | `ValueError` before integration |
| Non-finite `dt` | Python constructor and backend wrappers | `ValueError` before integration |
| Negative `noise_std` | Python constructor | `ValueError` before RNG use |
| Non-finite `noise_std` | Python constructor | `ValueError` before RNG use |
| Non-finite runtime current | Python `step()` | `ValueError` before state update |
| Empty RK4 current trace | backend wrappers | `ValueError` before backend execution |
| Non-1D RK4 current trace | Go, Julia, Mojo wrappers | `ValueError` before backend execution |
| Non-finite RK4 current trace | all maintained parity paths | `ValueError` before backend execution |
| Unknown RK4 model name | backend simulator | `ValueError` with unsupported-model message |

These guards are part of the production contract. They are not cosmetic input
checks: without them, a single invalid current or timestep can produce silent
NaN propagation through membrane state and benchmark traces.

## Numerical Notes

### Half-Euler default

The default path computes two half steps per public call:

| Quantity | Value |
|---|---:|
| Public default `dt` | 1.0 |
| Internal half step | 0.5 |
| RHS evaluations per call | 2 |
| Compatibility role | preserves historical public behaviour |

This path is intentionally retained for existing users and firing-pattern
tests that were calibrated against the historical model behaviour.

### RK4 path

The RK4 path evaluates the same two-dimensional right-hand side four times per
step:

| Quantity | Value |
|---|---:|
| State vector length | 2 |
| State keys | `v`, `u` |
| RHS evaluations per call | 4 |
| Noise handling | after deterministic integration |
| Threshold/reset handling | shared with half-Euler path |

RK4 is the maintained cross-language reference because it gives a stable,
deterministic state trace for Python/native parity comparisons.

### Floating-point parity interpretation

The benchmark distinguishes bit-exact equality from tolerance-bounded
scientific equality:

| Backend | Interpretation in latest Izhikevich run |
|---|---|
| Rust | bit-exact against Python for this trace |
| Julia | bit-exact against Python for this trace |
| Go | within tolerance, not bit-exact |
| Mojo | within tolerance, not bit-exact |

The non-bit-exact Go and Mojo outputs still satisfy the configured 1e-9
tolerance. Their maximum state delta was about 5.68e-14, which is far below
the acceptance threshold.

## Benchmark Reproduction Checklist

Before refreshing the published benchmark values:

1. Build the Rust engine wheel.
2. Install the rebuilt wheel into the active repository virtual environment.
3. Confirm the Rust RK4 simulator binding is importable.
4. Confirm Go, Julia, and Mojo wrapper availability.
5. Run the focused parity tests for every maintained backend.
6. Run the benchmark with a new timestamped JSON artifact.
7. Keep any existing benchmark artifact; do not delete old evidence.
8. Update the canonical `bench_neuron_integrators.json` only when the
   timestamped artifact for the same run exists.
9. Update this page with the artifact name, environment, parity result, and
   timing table.
10. Rebuild the docs in strict mode.

The append-only timestamped artifact is the evidence record. The canonical
JSON file is a convenience pointer for the latest local run.

## Documentation Contract

This page is considered current only when these fields are present:

| Required field | Present source |
|---|---|
| Source module and class | page header |
| Scientific reference | page header and references |
| Equations | `Scientific Model` |
| Defaults | `Parameters` |
| Integrator paths | `Integrator Paths` |
| Input contract | `Input Contract` |
| Backend contract | `Cross-Language RK4 Contract` |
| Latest benchmark artifact | `Latest Benchmark Evidence` |
| Timing table | `Izhikevich RK4 timing` |
| Parity table | `Izhikevich RK4 parity` |
| Test mapping | `Test Coverage` |
| Verification commands | `Verification Commands` |
| Scope boundary | `Known Scope Boundary` |

When any one of those fields becomes stale, refresh the measured evidence
instead of editing prose alone.

## Public API Example

```python
from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron

neuron = SCIzhikevichNeuron(
    a=0.02,
    b=0.2,
    c=-65.0,
    d=8.0,
    dt=0.5,
    noise_std=0.0,
)

spikes = []
for step_idx in range(1000):
    if neuron.step(10.0):
        spikes.append(step_idx)

state = neuron.get_state()
```

The same state shape is returned after both baseline and RK4 execution:

```python
assert set(state) == {"v", "u"}
```

## RK4 Benchmark API Example

```python
import numpy as np
from sc_neurocore_engine import py_rk4_neuron_simulate

currents = np.full(1000, 1.0, dtype=np.float64)
trace = py_rk4_neuron_simulate("izhikevich", currents, 1.0)

assert trace["n_steps"] == 1000
assert "v" in trace
assert "u" in trace
assert "spikes" in trace
```

Use this native path for parity and throughput checks. Use the public Python
class for ordinary application code unless the caller explicitly needs the
backend trace harness.

## Relationship to Other Documented Neurons

| Model | State variables | Spike mechanism | Maintained RK4 benchmark in latest run |
|---|---:|---|---|
| `SCIzhikevichNeuron` | 2 | quadratic membrane plus reset map | yes |
| `HodgkinHuxleyNeuron` | 4 | conductance dynamics and threshold crossing | yes |
| `AdExNeuron` | 2 | exponential spike initiation plus adaptation | yes |

The latest benchmark artifact contains all three models. This page reports the
Izhikevich rows only; the guide-level page reports the broader benchmark set.

## Evidence Freshness

| Evidence type | Latest artifact or command | Date |
|---|---|---|
| RK4 E2E benchmark | `bench_neuron_integrators_2026-05-09T1848.json` | 2026-05-09 |
| Rust wheel build | `target/wheels/sc_neurocore-3.14.0-cp312-cp312-manylinux_2_34_x86_64.whl` | 2026-05-09 |
| Pattern validation doc | `docs/validation/izhikevich_patterns.md` | 2026-03-28 record |
| Integrator guard tests | `tests/test_izhikevich_integrator_paths.py` | current tree |
| Backend parity tests | `tests/test_*_rk4_neuron_parity.py` | current tree |
| Docs strict build | `PYTHONPATH=src python -m mkdocs build --strict` | rerun after edits |

Benchmark values must not be copied forward after code changes unless the
benchmark has been rerun on the current tree and the new artifact is recorded.

## Known Scope Boundary

This page documents the bounded `SCIzhikevichNeuron` hardening and RK4 parity
evidence. It does not close the repository-wide audit item covering all neuron
models that still need upgraded integrator evidence, benchmarks, and public
performance documentation. That broader status remains tracked in
`docs/internal/AUDIT_INDEX.md`.

## References

1. Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE
   Transactions on Neural Networks*, 14(6), 1569-1572.
2. Izhikevich, E. M. (2004). Which model to use for cortical spiking neurons?
   *IEEE Transactions on Neural Networks*, 15(5), 1063-1070.
3. Izhikevich, E. M. (2007). *Dynamical Systems in Neuroscience: The Geometry
   of Excitability and Bursting*. MIT Press.
