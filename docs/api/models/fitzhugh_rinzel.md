# SPDX-License-Identifier: AGPL-3.0-or-later
# FitzHughRinzelNeuron

**Module:** `sc_neurocore.neurons.models.fitzhugh_rinzel`
**Rust engine:** `sc_neurocore_engine::neurons::simple_spiking::FitzHughRinzelNeuron`
**Polyglot mirrors:** Julia `FitzhughRinzelAccel`, Go `FitzHughRinzelNeuron`, Rust safety `FitzHughRinzelNeuron`
**Reference:** FitzHugh-Rinzel three-state qualitative bursting model after Rinzel's fast-slow classification work.
**Family:** cubic fast-slow burster: FitzHugh-Nagumo fast subsystem plus ultra-slow modulation.

---

## State and equations

The model evolves three continuous state variables:

| Variable | Role |
|----------|------|
| `v` | fast membrane-like activation |
| `w` | intermediate recovery |
| `y` | ultra-slow modulation / burst envelope |

The validated ODE contract is:

$$\frac{dv}{dt} = v - \frac{v^3}{3} - w + y + I$$

$$\frac{dw}{dt} = \delta(a + v - bw)$$

$$\frac{dy}{dt} = \mu(c - v - dy)$$

The public spike output is a threshold-crossing event:

$$v_{new} \geq v_{threshold} \land v_{old} < v_{threshold}$$

A spike resets the public state to `v=-1.0`, `w=-0.5`, `y=0.0`. This keeps the repository's spiking-neuron interface deterministic while preserving the continuous three-variable flow between threshold events.

---

## Numerical integration contract

All maintained implementations now use the same candidate-first fourth-order Runge-Kutta update over `(v, w, y)`:

1. Validate that mutable state, scalar parameters, current, and timestep are finite.
2. Require positive `b`, `d`, `delta`, `mu`, and `dt`.
3. Compute a full RK4 candidate without mutating public state.
4. Reject non-finite derivative or candidate values before mutation.
5. Commit the candidate and then apply threshold/reset semantics.

The RK4 candidate is equivalent to applying the standard four-stage integrator to the coupled ODE above, not to three independent scalar updates. The same contract is implemented in the Python reference, Rust engine benchmark path, Julia mirror, Go mirror, and Rust safety mirror.

Invalid runtime input fails closed:

| Condition | Python behavior | Julia / Go / Rust safety behavior | Rust engine behavior |
|-----------|-----------------|------------------------------------|----------------------|
| non-finite current | raises `FloatingPointError` before mutation | returns `0` and preserves state | returns `0` and preserves state |
| corrupted non-finite state | raises `FloatingPointError` before mutation | returns `0` and preserves state | returns `0` and preserves state |
| derivative overflow | raises `FloatingPointError` before mutation | returns `0` and preserves state | returns `0` and preserves state |
| non-finite candidate | raises `FloatingPointError` before mutation | returns `0` and preserves state | returns `0` and preserves state |

The language-specific error channel differs because Python exposes exceptions while the low-level service surfaces are non-throwing adapter paths.

---

## Parameters

| Parameter | Default | Contract | Description |
|-----------|---------|----------|-------------|
| `v` | `-1.0` | finite | fast variable |
| `w` | `-0.5` | finite | recovery variable |
| `y` | `0.0` | finite | ultra-slow modulation |
| `a` | `0.7` | finite | recovery nullcline offset |
| `b` | `0.8` | finite, positive | recovery nullcline slope |
| `c` | `-0.775` | finite | slow nullcline offset |
| `d` | `1.0` | finite, positive | slow nullcline slope |
| `delta` | `0.08` | finite, positive | intermediate timescale |
| `mu` | `0.0001` | finite, positive | ultra-slow timescale |
| `dt` | `0.1` | finite, positive | integration timestep |
| `v_threshold` | `1.0` | finite | spike event threshold |

The default timescale hierarchy is:

| Variable | Effective rate | Interpretation |
|----------|----------------|----------------|
| `v` | `1` | fast cubic spike dynamics |
| `w` | `delta = 0.08` | intermediate recovery, about 12.5 times slower than `v` |
| `y` | `mu = 0.0001` | ultra-slow envelope, about 10,000 times slower than `v` |

---

## Physics invariants covered by tests

Module-specific tests in `tests/test_model_fitzhugh_rinzel.py` assert the following contracts:

| Contract | Evidence |
|----------|----------|
| ODE formula | derivative helper equals the documented three-state RHS |
| RK4 fidelity | one-step update matches an independent RK4 reference to tight tolerance |
| finite-domain safety | invalid construction, invalid current, corrupted state, overflow, and non-finite candidates fail before mutation |
| slow-variable physics | changing `mu` changes the long-horizon `y` drift while preserving finite state |
| current regimes | moderate current produces repeated threshold events; quiescent and high-drive regimes remain deterministic |
| boundedness | long integrations remain finite and inside broad model-specific envelopes |
| reset semantics | spike threshold events reset `(v, w, y)` to the public spiking baseline |
| reproducibility | identical models under identical current sequences produce identical state and spike traces |
| public integration surfaces | population, network, projection, and analysis paths preserve the model contract |

The focused module evidence on 2026-05-31 was:

```text
PYTHONPATH=src .venv/bin/python -m coverage run --rcfile=/dev/null --source=src/sc_neurocore/neurons/models -m pytest tests/test_model_fitzhugh_rinzel.py -q
42 passed
src/sc_neurocore/neurons/models/fitzhugh_rinzel.py: 100% statement coverage
```

Polyglot and engine checks on the same hardening pass:

```text
julia --project=. -e 'include("src/sc_neurocore/accel/julia/neurons/fitzhugh_rinzel.jl"); ...'
go test src/sc_neurocore/accel/go/services/fitzhugh_rinzel.go
rustc --test src/sc_neurocore/accel/rust/safety/fitzhugh_rinzel.rs -o "$tmp/fhr_safety_test" && "$tmp/fhr_safety_test"
cargo test --manifest-path engine/Cargo.toml fhr_ -- --nocapture
```

Observed results: Julia valid-step check passed, Go compile/test passed, Rust safety tests passed with 5 tests, and Rust engine FHR tests passed with 8 tests.

---

## Benchmark evidence

Benchmark artefacts are stored under `benchmarks/results/`.

| Surface | Command | Result |
|---------|---------|--------|
| Python reference | `FitzHughRinzelNeuron.step(0.5)`, 7 repeats of 100,000 steps | median `1.1091808029450476e-05` seconds per step, deterministic 228 spikes per repeat |
| Rust engine | `cargo bench --manifest-path engine/Cargo.toml --bench full_bench fitzhugh_rinzel_10k_steps -- --sample-size 10` | Criterion estimate `572.65 µs` per 10k steps, `57.265 ns` per step |

The RK4 path is intentionally slower than the previous Euler benchmark because it evaluates the coupled ODE four times per step and validates the candidate before mutation.

---

## Relationship to FitzHugh-Nagumo

The fast subsystem is the FitzHugh-Nagumo cubic oscillator with an added slow bias term:

| Model | Fast equation |
|-------|---------------|
| FitzHugh-Nagumo | `dv/dt = v - v^3/3 - w + I` |
| FitzHugh-Rinzel | `dv/dt = v - v^3/3 - w + y + I` |

The slow `y` variable acts as an endogenous current modulation. When `y` increases, the effective drive to the fast subsystem increases; when `y` decreases, the fast subsystem moves toward quiescence. This is the minimal qualitative mechanism the project uses for three-state burst-envelope dynamics.
