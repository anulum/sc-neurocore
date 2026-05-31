# SPDX-License-Identifier: AGPL-3.0-or-later
# PernarowskiNeuron

**Module:** `sc_neurocore.neurons.models.pernarowski`
**Rust engine:** `sc_neurocore_engine::neurons::simple_spiking::PernarowskiNeuron`
**Polyglot mirrors:** Julia `PernarowskiAccel`, Go `PernarowskiNeuronState`, Rust safety `PernarowskiNeuron`
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
julia --project=. -e 'include("src/sc_neurocore/accel/julia/neurons/pernarowski.jl"); ...'
go test src/sc_neurocore/accel/go/services/pernarowski.go
rustc --test src/sc_neurocore/accel/rust/safety/pernarowski.rs -o "$tmp/pernarowski_safety_test" && "$tmp/pernarowski_safety_test"
cargo test --manifest-path engine/Cargo.toml pernarowski -- --nocapture
```

Observed results: Julia valid-step check passed, Go compile/test passed, Rust safety tests passed with 5 tests, and Rust engine Pernarowski tests passed with 9 tests.

---

## Benchmark evidence

Benchmark artefacts are stored under `benchmarks/results/`.

| Surface | Command | Result |
|---------|---------|--------|
| Python reference | `PernarowskiNeuron.step(0.5)`, 7 repeats of 100,000 steps | median `1.0787754809716717e-05` seconds per step, deterministic 343 spikes per repeat |
| Rust engine | `cargo bench --manifest-path engine/Cargo.toml --bench full_bench pernarowski_10k_steps -- --sample-size 10` | Criterion estimate `632.87 µs` per 10k steps, `63.287 ns` per step |

The RK4 path is slower than the prior Euler table value because each step evaluates the coupled ODE four times and validates the candidate before mutation.

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
