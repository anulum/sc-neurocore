# MATNeuron

**Module:** `sc_neurocore.neurons.models.mat`<br>
**Reference:** Kobayashi et al. 2009<br>
**Family:** Integrate-and-fire with multi-timescale adaptive threshold<br>
**State variables:** membrane voltage `v`, fast threshold component `theta1`,
slow threshold component `theta2`

`MATNeuron` implements the Kobayashi multi-timescale adaptive-threshold model as
a deterministic three-state integrate-and-fire neuron. The model separates
subthreshold membrane relaxation from two spike-triggered threshold memories:
`theta1` captures short refractory adaptation, while `theta2` accumulates slower
history-dependent rate adaptation.

## Equations

The maintained implementation advances the continuous state
`(v, theta1, theta2)` with one candidate-first RK4 step under piecewise-constant
input current for each timestep:

$$
\frac{dV}{dt} =
\frac{-(V - V_{rest}) + R I}{\tau_m}
$$

$$
\frac{d\theta_1}{dt} = -\frac{\theta_1}{\tau_1}
$$

$$
\frac{d\theta_2}{dt} = -\frac{\theta_2}{\tau_2}
$$

The effective spike threshold is evaluated against the RK4 candidate:

$$
V_{th}(t + \Delta t) =
V_{th,base} + \theta_1(t + \Delta t) + \theta_2(t + \Delta t)
$$

If the candidate voltage crosses that threshold, the voltage is reset and the
decayed candidate threshold state is retained before adding spike increments:

$$
V \leftarrow V_{reset}
$$

$$
\theta_1 \leftarrow \theta_1(t + \Delta t) + h_1
$$

$$
\theta_2 \leftarrow \theta_2(t + \Delta t) + h_2
$$

This replaces the older split update that used a forward-Euler voltage increment
followed by exact threshold decay. The RK4 contract keeps the membrane and both
threshold states under one integration boundary and prevents partial state
mutation when a candidate is invalid.

## Parameters

| Parameter | Default | Contract | Description |
|-----------|--------:|----------|-------------|
| `v` | -70.0 | finite, -200 to 100 | Initial membrane voltage. |
| `theta1` | 0.0 | finite, non-negative | Fast adaptive-threshold component. |
| `theta2` | 0.0 | finite, non-negative | Slow adaptive-threshold component. |
| `v_rest` | -70.0 | finite | Resting voltage. |
| `v_reset` | -70.0 | finite, -200 to 100 | Voltage after a spike. |
| `v_threshold_base` | -50.0 | finite | Baseline threshold before adaptation. |
| `tau_m` | 10.0 | positive finite | Membrane time constant. |
| `tau_1` | 10.0 | positive finite | Fast adaptation time constant. |
| `tau_2` | 200.0 | positive finite | Slow adaptation time constant. |
| `h1` | 5.0 | finite, non-negative | Fast threshold increment on spike. |
| `h2` | 3.0 | finite, non-negative | Slow threshold increment on spike. |
| `resistance` | 1.0 | positive finite | Input-current scaling. |
| `dt` | 1.0 | positive finite | Integration timestep. |

## Runtime Contract

- Constructor validation rejects non-finite state, non-positive time constants,
  non-positive timestep or resistance, negative adaptation state, and voltages
  outside the bounded safety envelope.
- `step(current)` rejects non-finite runtime current before evaluating RK4.
- Candidate state is computed before mutation. If any RK4 candidate is non-finite
  or leaves the bounded voltage/adaptation envelope, the Python path raises and
  non-throwing native mirrors return `-1` while preserving previous state.
- Spike handling commits the decayed RK4 threshold candidates plus `h1` and `h2`;
  it does not discard substep threshold decay.
- `reset()` restores `v` to `v_rest` and clears both threshold components.

## Polyglot Surfaces

| Surface | Path | Contract |
|---------|------|----------|
| Python reference | `src/sc_neurocore/neurons/models/mat.py` | Stateful RK4 implementation with Python exceptions on invalid contracts. |
| Go service | `src/sc_neurocore/accel/go/services/mat.go` | Stateful RK4 mirror; returns `-1` on invalid input/state without mutation. |
| Julia mirror | `src/sc_neurocore/accel/julia/neurons/mat.jl` | Stateful RK4 mirror; returns `-1` on invalid input/state without mutation. |
| Mojo kernel | `src/sc_neurocore/accel/mojo/kernels/mat.mojo` | Stateless RK4 helper surface for accelerator integration. |
| Rust safety | `src/sc_neurocore/accel/rust/safety/mat.rs` | Stateful RK4 safety mirror; returns `-1` on invalid input/state without mutation. |

## Measured Benchmark Evidence

Benchmark artifact:
`benchmarks/results/local_python_2026-06-18_mat_rk4.json`

Evidence class: `local_regression_non_isolated`. The run measures local
regression parity only; it is not an isolated hardware-performance claim and does
not claim production speed.

Configuration:

| Field | Value |
|-------|------:|
| Steps per repeat | 200,000 |
| Repeats | 5 |
| Constant current | 50.0 |
| Expected spike parity | 8,620 spikes on every backend |

Measured timing on `aaarthuus`, Linux 6.17.0-35-generic, Python 3.12.3:

| Backend | Median ns/step | Min ns/step | Max ns/step | Spikes |
|---------|---------------:|------------:|------------:|-------:|
| Python | 2,868.59375 | 2,825.063975 | 2,871.083915 | 8,620 |
| Rust safety | 30.195125 | 30.016825 | 30.48844 | 8,620 |
| Go service | 39.51 | 39.17 | 40.48 | 8,620 |
| Julia mirror | 36.70748 | 36.263415 | 37.02197 | 8,620 |
| Mojo helper | 27.71603496512398 | 27.55250985501334 | 27.785264828708023 | 8,620 |

The benchmark gate
`mat-rk4-multibackend-local-regression` requires numeric timing rows for all five
backends and zero-tolerance spike-count parity across Python, Rust, Go, Julia,
and Mojo.

## Verification

Focused checks for this hardening slice:

| Command | Coverage |
|---------|----------|
| `ruff check src/sc_neurocore/neurons/models/mat.py tests/test_model_mat.py benchmarks/bench_model_mat.py` | Python lint for the model, tests, and benchmark. |
| `mypy --strict src/sc_neurocore/neurons/models/mat.py benchmarks/bench_model_mat.py` | Strict typing for the Python model and benchmark harness. |
| `pytest tests/test_model_mat.py -q` | Python RK4 math, validation, network, analysis, and performance contract tests. |
| `go test src/sc_neurocore/accel/go/services/mat.go src/sc_neurocore/accel/go/services/mat_test.go` | Go RK4 candidate, spike, invalid-state, and benchmark hook coverage. |
| `rustc --test src/sc_neurocore/accel/rust/safety/mat.rs -o /tmp/mat_safety_test && /tmp/mat_safety_test` | Rust safety RK4 and fail-closed coverage. |
| `julia --project=. -e '<MAT RK4 smoke>'` | Julia RK4 candidate and invalid-state preservation smoke. |
| `mojo -I src/sc_neurocore/accel/mojo/kernels <MAT smoke>` | Mojo stateless RK4 helper smoke. |
| `python benchmarks/bench_model_mat.py` | Regenerates the five-backend local benchmark artifact. |
| `python tools/benchmark_evidence_gate.py --manifest /tmp/mat_gate.json --output /tmp/mat_gate_report.json` | Validates benchmark schema, required timing rows, source hashes, and spike parity. |

## Usage

```python
from sc_neurocore.neurons.models.mat import MATNeuron

neuron = MATNeuron()
spikes = [neuron.step(50.0) for _ in range(1000)]
print(sum(spikes), neuron.v, neuron.theta1, neuron.theta2)
```

Invalid runtime current is rejected before state mutation:

```python
neuron = MATNeuron()
before = (neuron.v, neuron.theta1, neuron.theta2)
try:
    neuron.step(float("nan"))
except ValueError:
    assert (neuron.v, neuron.theta1, neuron.theta2) == before
```
