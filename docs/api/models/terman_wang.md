# SPDX-License-Identifier: AGPL-3.0-or-later
# TermanWangOscillator

**Module:** `sc_neurocore.neurons.models.terman_wang`
**Rust engine:** `sc_neurocore_engine::neurons::simple_spiking::TermanWangOscillator`
**Polyglot mirrors:** Julia `TermanWangAccel`, Go `TermanWangOscillatorState`, Mojo kernel notes, Rust safety `TermanWangOscillator`
**Reference:** Terman, D. & Wang, D. L. (1995). Neural Computation, 7(5), 1035-1064.
**Family:** two-state relaxation oscillator for LEGION-style temporal segmentation.

---

## Equations

The maintained model evolves two continuous state variables:

| Variable | Role |
|----------|------|
| `v` | fast excitatory variable |
| `w` | slow inhibitory recovery variable |

The ODE contract is:

$$\frac{dv}{dt} = f(v) - w + I + \rho$$

$$\frac{dw}{dt} = \epsilon(g(v) - w)$$

with:

$$f(v) = 3v - v^3 + 2$$

$$g(v) = \alpha(1 + \tanh(v / \beta))$$

The public spike output is an upward threshold-crossing event:

$$v_{new} \geq v_{peak} \land v_{old} < v_{peak}$$

`step()` does not reset after a spike. The oscillator remains on the continuous Terman-Wang relaxation trajectory; `reset()` is an explicit caller action only.

---

## Numerical integration contract

Python, Rust engine, Julia, Go, Mojo notes, and Rust safety surfaces now use candidate-first RK4 integration over the coupled `(v, w)` system:

1. Validate finite state, finite scalar parameters, positive `beta`, positive `epsilon`, positive `dt`, finite threshold, and finite external drive.
2. Evaluate all four RK4 derivative stages against the documented cubic/sigmoid ODE.
3. Reject derivative overflow, non-finite derivative output, and non-finite candidate state before mutation.
4. Commit `(v, w)` only after candidate validation.
5. Emit a spike only for upward threshold crossings.

Invalid runtime input fails closed:

| Condition | Python behavior | Julia behavior | Go / Rust safety behavior | Rust engine behavior |
|-----------|-----------------|----------------|---------------------------|----------------------|
| non-finite current | raises `FloatingPointError` before mutation | returns `-1` and preserves state | returns error and preserves state | returns `0` and preserves state |
| non-scalar current | raises `TypeError` before mutation | adapter type boundary rejects non-float input | language type boundary rejects non-float input | Rust type boundary rejects non-float input |
| corrupted non-finite state | raises `FloatingPointError` before mutation | returns `-1` and preserves state | returns error and preserves state | returns `0` and preserves state |
| non-positive `beta`, `epsilon`, or `dt` | raises `ValueError` before mutation | returns `-1` and preserves state | returns error and preserves state | returns `0` and preserves state |
| derivative overflow or non-finite candidate | raises `FloatingPointError` before mutation | returns `-1` and preserves state | returns error and preserves state | returns `0` and preserves state |

---

## Parameters

| Parameter | Default | Contract | Description |
|-----------|---------|----------|-------------|
| `v` | `-1.5` | finite | fast excitatory state |
| `w` | `-0.5` | finite | slow recovery state |
| `alpha` | `3.0` | finite | recovery sigmoid amplitude |
| `beta` | `0.2` | finite, positive | recovery sigmoid steepness |
| `epsilon` | `0.02` | finite, positive | slow recovery timescale ratio |
| `rho` | `0.0` | finite | tonic inhibitory/bias term |
| `dt` | `0.05` | finite, positive | integration timestep |
| `v_peak` | `1.5` | finite | spike event threshold |

`epsilon = 0.02` makes `w` approximately 50 times slower than the fast cubic voltage state, producing relaxation oscillations with sharp fast jumps and slow recovery drift.

---

## Behavioural evidence

Module-specific tests in `tests/test_model_terman_wang.py` assert:

| Contract | Evidence |
|----------|----------|
| ODE formula | derivative helper equals the documented cubic/sigmoid RHS |
| RK4 fidelity | one-step update matches an independent RK4 reference |
| slow recovery | single-step `v` movement remains much larger than `w` movement |
| oscillatory regimes | moderate drive produces repeated threshold crossings; zero/high drive remains bounded or suppressed |
| finite-domain safety | invalid construction, invalid current, corrupted state, invalid runtime scales, derivative overflow, and non-finite candidates fail before mutation |
| reproducibility | identical current traces produce identical state and spike traces |
| public integration | population, network, projection, Poisson input, monitor, and spike-count analysis remain wired |

Focused evidence from 2026-05-31:

```text
PYTHONPATH=src .venv/bin/python -m coverage run --rcfile=/dev/null --source=src/sc_neurocore/neurons/models -m pytest tests/test_model_terman_wang.py -q
47 passed
src/sc_neurocore/neurons/models/terman_wang.py: 100% statement coverage
```

Polyglot and engine checks from the same pass:

```text
julia --project=. -e 'include("src/sc_neurocore/accel/julia/neurons/terman_wang.jl"); ...'
go test src/sc_neurocore/accel/go/services/terman_wang.go src/sc_neurocore/accel/go/services/terman_wang_test.go
rustc --test src/sc_neurocore/accel/rust/safety/terman_wang.rs -o "$tmp/terman_wang_safety_test" && "$tmp/terman_wang_safety_test"
cargo test --manifest-path engine/Cargo.toml tw_ -- --nocapture
```

Observed results: Julia valid-step check passed, Go tests passed, Rust safety tests passed with 6 tests, and Rust engine Terman-Wang tests passed with 7 tests.

---

## Benchmark evidence

Benchmark artefacts are stored under `benchmarks/results/`.

| Surface | Command | Result |
|---------|---------|--------|
| Python reference | `TermanWangOscillator.step(1.0)`, 7 repeats of 100,000 steps | median `9.56917773000896e-06` seconds per step, deterministic 29 spikes per repeat |
| Rust engine | `cargo bench --manifest-path engine/Cargo.toml --bench full_bench terman_wang_10k_steps -- --sample-size 10` | Criterion estimate `1.2380 ms` per 10k steps, `123.80 ns` per step |

The RK4 path is slower than the prior Euler table value because each step evaluates the cubic/sigmoid RHS four times and validates the candidate before mutation.

---

## Infrastructure pipeline

```text
TermanWangOscillator
├── Python reference: candidate-first RK4, continuous threshold-crossing event
├── Rust engine: candidate-first RK4 benchmark and production path
├── Julia mirror: candidate-first RK4 adapter path
├── Go mirror: candidate-first RK4 service path
├── Mojo kernel notes: RK4 candidate contract mirror
├── Rust safety mirror: candidate-first RK4 fail-closed path
├── Population / Network / Projection / Monitor integration
└── Spike-count analysis integration
```
