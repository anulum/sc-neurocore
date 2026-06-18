# RallCableNeuron

**Module:** `sc_neurocore.neurons.models.rall_cable`
**Reference:** Rall, W. (1959). *Experimental Neurology* 1:491-527
**Family:** passive multi-compartment cable
**Maintained surfaces:** Python, Rust engine, Rust safety, Go service, Julia kernel, Mojo fixed-five kernel

---

## Equations

The model stores one membrane potential per compartment. Current is injected at
the distal compartment and spikes are detected at the soma.

For compartment `i`:

$$
\tau_m \frac{dV_i}{dt} =
-(V_i - E_L) + g_r(V_{i-1} - 2V_i + V_{i+1}) + I_i
$$

`g_r` is `g_ratio`, and `I_i` is non-zero only at the distal end. The cable uses
sealed-end boundary conditions:

$$
V_{-1}=V_0,\qquad V_N=V_{N-1}
$$

The production step solves the passive cable operator with a candidate-first
implicit tridiagonal system. Let `u = V - E_L` and `alpha = dt / tau_m`.

$$
\left(I - \alpha(-I + g_r L)\right)u_{n+1} = u_n + \alpha I
$$

For interior compartments, the diagonal is `1 + alpha + 2 alpha g_ratio` and
the off-diagonals are `-alpha g_ratio`. At the sealed ends, the diagonal is
`1 + alpha + alpha g_ratio`. A one-compartment cable reduces to the scalar
leak solve with diagonal `1 + alpha`.

The step commits the solved candidate only if all candidate values are finite.
If the candidate soma crosses threshold from below, the soma is reset to
`v_reset`; dendritic compartments keep their solved values.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `n_comp` | `5` | compartments | Positive compartment count |
| `tau_m` | `20.0` | ms | Membrane time constant |
| `v_rest` | `-65.0` | mV | Leak reversal potential |
| `g_ratio` | `0.5` | dimensionless | Axial-to-leak coupling ratio |
| `v_threshold` | `-50.0` | mV | Somatic spike threshold |
| `v_reset` | `-65.0` | mV | Somatic reset after spike |
| `dt` | `0.1` | ms | Integration step |

Python raises `ValueError` for invalid construction, non-finite current, or
corrupt runtime state. Go, Rust safety, Julia, and Mojo return failure
sentinels or non-finite candidates without committing the invalid update.

---

## Numerical Contract

The implicit solve removes the old explicit-Euler stability dependency for the
linear passive cable step. It preserves the sealed-end Rall cable stencil,
keeps updates simultaneous across compartments, and avoids mutating state until
the tridiagonal solve succeeds.

Default parameters with `current=500.0` remain subthreshold at the soma because
the five-compartment passive cable attenuates the distal drive. Shorter cables
or higher coupling can still spike, and the test suite covers that regime.

---

## Polyglot Validation

| Surface | File | Validation |
|---------|------|------------|
| Python | `src/sc_neurocore/neurons/models/rall_cable.py` | `tests/test_model_rall_cable.py` |
| Rust engine | `engine/src/neurons/multi_compartment.rs` | targeted Cargo engine tests |
| Rust safety | `src/sc_neurocore/accel/rust/safety/rall_cable.rs` | `rustc --test` |
| Go service | `src/sc_neurocore/accel/go/services/rall_cable.go` | `rall_cable_test.go` |
| Julia kernel | `src/sc_neurocore/accel/julia/neurons/rall_cable.jl` | smoke step and benchmark |
| Mojo kernel | `src/sc_neurocore/accel/mojo/kernels/rall_cable.mojo` | smoke step and benchmark |

The benchmark requires Python, Rust engine, Go, Julia, and Mojo. It fails closed
if any maintained backend is unavailable.

---

## Local Benchmark Evidence

Artifact:
`benchmarks/results/local_python_2026-06-18_rall_cable_implicit.json`

Evidence class: `local_regression_non_isolated`

This is a local regression comparison on the development workstation. It is not
an isolated production hardware measurement and does not support a production
speed claim.

Command:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/bench_model_rall_cable.py
```

Configuration: `steps=200000`, `repeats=5`, `current=500.0`, `n_comp=5`.

| Backend | Median ns/step | Min ns/step | Max ns/step | Spikes |
|---------|----------------|-------------|-------------|--------|
| Python | 18633.773545 | 18126.659645 | 18857.107415 | 0 |
| Rust engine | 677.157000 | 653.795345 | 747.625365 | 0 |
| Go service | 173.400000 | 169.800000 | 180.100000 | 0 |
| Julia | 148.298740 | 146.888480 | 878.248485 | 0 |
| Mojo | 27.449250 | 27.203955 | 27.822800 | 0 |

All measured backends finish at the same soma and distal voltages:
`soma=-62.607655502392916`, `distal=301.0287081339704`.

The committed benchmark gate checks the artifact contract, finite timing
metrics, exact spike-count parity, and source hashes for the Python model,
Rust engine example, Rust engine source, Go, Julia, Mojo, Rust safety, and
benchmark harness.

---

## API

### `RallCableNeuron.step(current: float) -> int`

Advance one implicit passive cable step and return `1` only when the soma
crosses threshold from below.

### `RallCableNeuron.reset() -> None`

Reset every compartment to `v_rest`.

### `RallCableNeuron.v`

NumPy array of compartment voltages. `v[0]` is soma, `v[-1]` is distal.
