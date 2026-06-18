# SRM0Neuron

**Module:** `sc_neurocore.neurons.models.srm0`
**Reference:** Gerstner & Kistler, *Spiking Neuron Models*, Cambridge University Press, 2002, Ch. 4
**Family:** Spike Response Model, zeroth order
**Maintained surfaces:** Python, Rust safety, Go service, Julia kernel, Mojo kernel

---

## Equations

The implemented SRM0 state is the membrane potential `v`, refractory kernel
`eta`, and internal time `t`. For one step of duration `dt` with constant input
current `I`, the continuous dynamics are:

$$
\frac{d\eta}{dt} = -\frac{\eta}{\tau_\eta}
$$

$$
\frac{dv}{dt} =
\frac{V_{rest} + R I + \eta(t) - v}{\tau_m}
$$

The production update uses the exact coupled flow over the step, not a forward
Euler membrane approximation:

$$
\eta_{n+1} = \eta_n e^{-dt/\tau_\eta}
$$

$$
v_{n+1} =
V_\infty + (v_n - V_\infty)e^{-dt/\tau_m}
+ \eta_n C(dt, \tau_m, \tau_\eta)
$$

where:

$$
V_\infty = V_{rest} + R I
$$

$$
C =
\frac{e^{-dt/\tau_\eta} - e^{-dt/\tau_m}}
{\tau_m \left(\frac{1}{\tau_m} - \frac{1}{\tau_\eta}\right)}
$$

When `tau_m == tau_eta`, the coupling term uses the analytic limit:

$$
C = \frac{dt}{\tau_m} e^{-dt/\tau_m}
$$

If `v_next >= v_threshold`, the step emits one spike, commits `v_rest`, sets
`eta = -eta_reset`, advances time by `dt`, and records the last spike time.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | `0.0` | a.u. | Initial membrane potential |
| `v_rest` | `0.0` | a.u. | Resting potential |
| `v_threshold` | `1.0` | a.u. | Spike threshold |
| `tau_m` | `20.0` | ms | Membrane time constant |
| `tau_eta` | `50.0` | ms | Refractory-kernel time constant |
| `eta_reset` | `5.0` | a.u. | Positive reset amplitude applied as negative refractory offset |
| `resistance` | `1.0` | a.u./current | Input gain |
| `dt` | `1.0` | ms | Integration step |

All time constants, `dt`, `eta_reset`, and `resistance` must be finite and
positive. Python raises `ValueError` for invalid parameters or non-finite input
current. The lower-level Rust, Go, Julia, and Mojo surfaces return a failure
sentinel or non-finite candidate value according to their language boundary.

---

## Numerical Contract

The exact-flow update removes timestep-dependent membrane Euler error for the
piecewise-constant-current SRM0 step while retaining the standard
`step(current) -> int` spiking interface.

The refractory kernel enters as a time-varying offset to the effective resting
potential. After a spike, `eta = -eta_reset`; it decays exponentially toward
zero and temporarily increases the current needed for another spike. This is
graded refractoriness rather than an absolute refractory clamp.

The Python surface is the public model implementation. The Rust safety, Go,
Julia, and Mojo surfaces implement the same step equation for parity tests and
benchmark evidence.

---

## Polyglot Validation

The SRM0 exact-flow slice is covered by:

| Surface | File | Validation |
|---------|------|------------|
| Python | `src/sc_neurocore/neurons/models/srm0.py` | `tests/test_model_srm0.py` |
| Rust safety | `src/sc_neurocore/accel/rust/safety/srm0.rs` | `rustc --test` |
| Go service | `src/sc_neurocore/accel/go/services/srm0.go` | `srm0_test.go` |
| Julia kernel | `src/sc_neurocore/accel/julia/neurons/srm0.jl` | smoke step and benchmark |
| Mojo kernel | `src/sc_neurocore/accel/mojo/kernels/srm0.mojo` | smoke step and benchmark |

The comparison benchmark requires all five maintained surfaces. It fails closed
if any backend is missing.

---

## Local Benchmark Evidence

Artifact:
`benchmarks/results/local_python_2026-06-18_srm0_exact_flow.json`

Evidence class: `local_regression_non_isolated`

This run is a local regression comparison on the development workstation. It is
not an isolated production hardware measurement and does not support a
production speed claim.

Command:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/bench_model_srm0.py
```

Configuration: `steps=200000`, `repeats=5`, `current=2.0`.

| Backend | Median ns/step | Min ns/step | Max ns/step | Spikes |
|---------|----------------|-------------|-------------|--------|
| Python | 1598.465110 | 1519.138290 | 1641.185615 | 1905 |
| Rust safety | 19.277575 | 18.721155 | 19.369475 | 1905 |
| Go service | 29.540000 | 28.850000 | 30.560000 | 1905 |
| Julia | 23.176310 | 22.518675 | 23.896565 | 1905 |
| Mojo | 6.995495 | 6.984085 | 7.047535 | 1905 |

The committed benchmark gate checks:

- artifact SPDX marker,
- benchmark name,
- `evidence_class = local_regression_non_isolated`,
- `production_speed_claim = false`,
- `hardware_measurement_claimed = false`,
- finite timing and spike-count metrics for every backend,
- exact spike-count parity across Python, Rust, Go, Julia, and Mojo,
- source hashes for each maintained implementation and the benchmark harness.

---

## API

### `SRM0Neuron.step(current: float) -> int`

Advance one exact-flow step with constant current over `dt`.

Returns `1` when the candidate membrane crosses threshold and a spike is
committed; otherwise returns `0`.

### `SRM0Neuron.reset() -> None`

Reset `v` to `v_rest`, clear `eta`, reset the internal clock, and clear the
last spike timestamp.

### `SRM0Neuron.get_state() -> dict[str, float]`

Return `{"v": v, "eta": eta, "t": t}` for trace inspection and parity tests.
