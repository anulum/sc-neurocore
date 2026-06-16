# SPDX-License-Identifier: AGPL-3.0-or-later
# PernarowskiNeuron

**Module:** `sc_neurocore.neurons.models.pernarowski`
**Rust engine:** `sc_neurocore_engine::neurons::simple_spiking::PernarowskiNeuron`
**Polyglot `simulate` backends:** Rust engine (PyO3), Julia `PernarowskiAccel`, Go c-shared (`accel/go/neurons/pernarowski`), Mojo FFI (`accel/mojo/neurons/pernarowski.mojo`); standalone Rust safety mirror `PernarowskiNeuron`
**Reference:** Pernarowski, M. (1994). SIAM Journal on Applied Mathematics, 54, 814-832.
**Family:** three-state beta-cell burster with fast cubic voltage and two slower recovery/adaptation variables.

---

## Equations

The maintained model evolves three continuous state variables:

| Variable | Role |
|----------|------|
| `v` | fast voltage-like cubic state |
| `w` | intermediate recovery variable |
| `z` | ultra-slow adaptation variable |

The ODE contract is:

$$\frac{dv}{dt} = v - \frac{v^3}{3} - w - z + I$$

$$\frac{dw}{dt} = \epsilon_1(v - \gamma w + \alpha)$$

$$\frac{dz}{dt} = \epsilon_2(\beta(v + 0.7) - z)$$

The public spike output is an upward threshold-crossing event:

$$v_{new} \geq v_{threshold} \land v_{old} < v_{threshold}$$

`step()` does not reset the state after a spike. The trajectory remains the continuous Pernarowski flow; `reset()` is an explicit caller action only.

---

## Numerical integration contract

Python, Rust engine, Julia, Go, and Rust safety surfaces now use the same candidate-first RK4 update over the coupled `(v, w, z)` system:

1. Validate finite state, finite offsets, finite threshold, positive timescale/coupling parameters, and finite current.
2. Evaluate all four RK4 derivative stages against the same coupled ODE.
3. Reject derivative overflow, non-finite derivative output, and non-finite candidate state before mutation.
4. Commit `(v, w, z)` only after candidate validation.
5. Emit a spike only for upward threshold crossings, without artificial reset.

Invalid runtime input fails closed:

| Condition | Python behavior | Julia / Go / Rust safety behavior | Rust engine behavior |
|-----------|-----------------|------------------------------------|----------------------|
| non-finite current | raises `FloatingPointError` before mutation | returns `0` and preserves state | returns `0` and preserves state |
| non-scalar current | raises `TypeError` before mutation | adapter type boundary rejects non-float input | Rust type boundary rejects non-float input |
| corrupted non-finite state | raises `FloatingPointError` before mutation | returns `0` and preserves state | returns `0` and preserves state |
| non-positive timescale/coupling | raises `ValueError` before mutation | returns `0` and preserves state | returns `0` and preserves state |
| derivative overflow or non-finite candidate | raises `FloatingPointError` before mutation | returns `0` and preserves state | returns `0` and preserves state |

---

## Parameters

| Parameter | Default | Contract | Description |
|-----------|---------|----------|-------------|
| `v` | `-1.0` | finite | fast voltage-like state |
| `w` | `0.0` | finite | intermediate recovery state |
| `z` | `0.0` | finite | ultra-slow adaptation state |
| `alpha` | `0.1` | finite | recovery nullcline offset |
| `beta` | `0.5` | finite | adaptation nullcline slope |
| `eps1` | `0.1` | finite, positive | intermediate timescale ratio |
| `eps2` | `0.001` | finite, positive | ultra-slow timescale ratio |
| `gamma` | `0.5` | finite, positive | recovery self-coupling |
| `dt` | `0.1` | finite, positive | integration timestep |
| `v_threshold` | `0.5` | finite | spike event threshold |

The default hierarchy keeps `z` about 100 times slower than `w`, so `z` modulates the burst envelope rather than the individual spike timescale.

---

## Behavioural evidence

Module-specific tests in `tests/test_model_pernarowski.py` assert:

| Contract | Evidence |
|----------|----------|
| ODE formula | derivative helper equals the documented three-state RHS |
| RK4 fidelity | one-step update matches an independent RK4 reference |
| continuous threshold semantics | spikes occur only on upward threshold crossings, without implicit reset |
| bounded voltage | long-horizon traces remain finite and inside broad cubic-nullcline envelopes |
| slow-variable separation | `z` evolves much more slowly than `w`; changing `eps2` changes dynamics |
| current regimes | moderate current sustains oscillation; high drive enters depolarisation block |
| finite-domain safety | invalid construction, invalid current, corrupted state, invalid runtime scales, derivative overflow, and non-finite candidates fail before mutation |
| public integration | population, network, monitor, Poisson input, and spike-count analysis contracts remain wired |

Focused evidence from 2026-05-31:

```text
PYTHONPATH=src .venv/bin/python -m coverage run --rcfile=/dev/null --source=src/sc_neurocore/neurons/models -m pytest tests/test_model_pernarowski.py -q
87 passed
src/sc_neurocore/neurons/models/pernarowski.py: 100% statement coverage
```

Polyglot and engine checks from the same pass:

```text
cargo test --manifest-path engine/Cargo.toml pernarowski -- --nocapture
pytest tests/test_pernarowski_backends.py
```

Observed results: Rust engine Pernarowski tests pass (9 tests); the cross-backend
parity suite confirms Rust/Julia/Go bit-exactness and the Mojo ULP band.

---

## Polyglot acceleration

A single `step` is trivial, but an N-step run is a sequential RK4 recurrence that
does not vectorise, so a compiled inner loop genuinely beats Python.
`simulate(n_steps, current, backend="auto")` dispatches across the polyglot chain
and returns `(trace, spikes)`:

```python
from sc_neurocore.neurons.models.pernarowski import PernarowskiNeuron

neuron = PernarowskiNeuron()
trace, spikes = neuron.simulate(20_000, current=0.0)   # auto → Rust
```

The right-hand side is exact polynomial floating-point arithmetic — the cubic is
written `v*v*v` so it matches the engine's `v.powi(3)` to the last bit, with no
transcendental functions — so **Rust, Julia and Go reproduce the NumPy reference
bit-for-bit**. Mojo's release build contracts the RK4 multiply-adds into fused
multiply-adds; the model is a periodic slow-fast burster (not chaotic), so the
single-ULP difference stays bounded over millions of steps and the spike counts
match. `auto` selects Rust (the fastest bit-exact backend, shipped in the wheel).

### Measured throughput

2,000,000 RK4 steps, default bursting regime (`current=0.0`), median of 5 repeats.
Non-isolated loaded workstation (Intel i5-11600K) per
`BROADCAST_2026-06-04_benchmark_core_isolation` — functional/regression evidence,
not an isolated-core figure. Reproduce with
`python benchmarks/bench_pernarowski_simulate.py`.

| Backend | Median (ms) | Speed-up vs Python | Whole-trace parity |
|---------|------------:|-------------------:|--------------------|
| python  | 2224.59 | 1.0× | reference |
| mojo    | 102.34 | 21.7× | 1.6×10⁻¹² (FMA, non-amplifying) |
| go      | 109.63 | 20.3× | bit-exact (0) |
| julia   | 112.26 | 19.8× | bit-exact (0) |
| rust (`auto`) | 124.82 | 17.8× | bit-exact (0) |

Artefact: `benchmarks/results/bench_pernarowski_simulate.json`. The earlier
single-language step-level criterion/Python figures remain valid regression
evidence for the per-step `step` path.

---

## Infrastructure pipeline

```text
PernarowskiNeuron
├── Python reference: candidate-first RK4, continuous threshold-crossing event
├── Rust engine: candidate-first RK4 benchmark and production path
├── Julia mirror: candidate-first RK4 adapter path
├── Go mirror: candidate-first RK4 service path
├── Rust safety mirror: candidate-first RK4 fail-closed path
├── Population / Network / Monitor integration
└── Spike-count analysis integration
```
