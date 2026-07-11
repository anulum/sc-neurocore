<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# FitzHughRinzelNeuron

**Module:** `sc_neurocore.neurons.models.fitzhugh_rinzel`
**Rust engine:** `sc_neurocore_engine::neurons::simple_spiking::FitzHughRinzelNeuron`
**Polyglot `simulate` chain (RK4):** Rust (`py_fitzhugh_rinzel_simulate`), Julia (`FitzHughRinzelAccel`), Go (`accel/go/neurons/fitzhugh_rinzel`, c-shared), Mojo (`accel/mojo/neurons/fitzhugh_rinzel.mojo`) — see *Polyglot acceleration* below
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

The event does **not** reset any state. The continuous three-variable flow remains
intact, and a new event is reported only after `v` falls below the threshold and
crosses upward again.

---

## Numerical integration contract

All maintained implementations now use the same candidate-first fourth-order Runge-Kutta update over `(v, w, y)`:

1. Validate that mutable state, scalar parameters, current, and timestep are finite.
2. Require positive `b`, `d`, `delta`, `mu`, and `dt`.
3. Compute a full RK4 candidate without mutating public state.
4. Reject non-finite derivative or candidate values before mutation.
5. Commit the candidate and report the rising-edge threshold decision without a reset.

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
| edge semantics | a spike is reported only on an upward threshold crossing and leaves `(v, w, y)` unchanged |
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

## Schema-to-RTL co-simulation

The bundled `fitzhugh_rinzel` TOML and JSON schemas mirror the maintained Python
contract: three coupled state variables, simultaneous classical RK4 at `dt=0.1`,
the exact `v * v * v` operation order, rising-edge `v >= v_threshold` detection,
and no reset. At the enrolled 3000-step operating point (`I=0.5`), the hand model,
the schema runner, and the emitted Q16.16 RTL each report eight crossings.

The exact spike-count result also holds across the tested `I=0.4` to `I=0.6`
band (seven, eight, and eight crossings). `I=0.7` is deliberately outside the
contract: a marginal ninth crossing moves across the threshold under fixed-point
rounding. This boundary is recorded rather than hidden behind a broad tolerance.

The committed `fitzhugh_rinzel_driven_bursting_doi` trace independently re-derives
the three-state RK4 recurrence and checks spike count, first-spike step, and the
final/minimum/maximum/mean of `v`, `w`, and `y`. Its provenance is Rinzel's 1987
*A Formal Classification of Bursting Mechanisms in Excitable Systems*, DOI
`10.1007/978-3-642-93360-8_26`.

Focused evidence:

```text
tests/test_cosimulation.py::TestQ1616Precision::test_fitzhugh_rinzel_q1616_parity
tests/test_reference_traces.py::test_trace_features_match_independent_reference[fitzhugh_rinzel]
```

---

## Benchmark evidence

Benchmark artefacts are stored under `benchmarks/results/`.

| Surface | Command | Result |
|---------|---------|--------|
| Python reference | `FitzHughRinzelNeuron.step(0.5)`, 7 repeats of 100,000 steps | median `1.1091808029450476e-05` seconds per step, deterministic 228 spikes per repeat |
| Rust engine | `cargo bench --manifest-path engine/Cargo.toml --bench full_bench fitzhugh_rinzel_10k_steps -- --sample-size 10` | Criterion estimate `572.65 µs` per 10k steps, `57.265 ns` per step |

The RK4 path is intentionally slower than the previous Euler benchmark because it evaluates the coupled ODE four times per step and validates the candidate before mutation.

---

## Polyglot acceleration

`step` runs one RK4 update, but `simulate(n_steps, current, backend=...)` is a
sequential recurrence (each step depends on the previous) that does not
vectorise — a compiled inner loop genuinely beats Python. The kernel carries a
full polyglot chain over the RK4 integrator (the only integrator this model
exposes):

```python
from sc_neurocore.neurons.models.fitzhugh_rinzel import FitzHughRinzelNeuron

neuron = FitzHughRinzelNeuron()
trace, spikes = neuron.simulate(2_000_000, current=0.5)           # auto -> Rust
trace, spikes = neuron.simulate(2_000_000, 0.5, backend="go")    # force a backend
```

`backend` accepts `"auto" | "rust" | "julia" | "go" | "mojo" | "python"`. `auto`
prefers Rust (it ships in the `sc_neurocore_engine` wheel). `trace[t]` is `v`
after step `t`; `spikes` counts upward crossings of `v_threshold`.

The RK4 right-hand side is **exact arithmetic** — the cube is written `v*v*v`
(bit-identical to Rust `v.powi(3)`, Julia `v^3` and Go/Mojo `v*v*v`), with no
transcendental functions. So **Rust, Julia and Go reproduce the NumPy trace
bit-for-bit**, verified over a 60,000-step slow-burst run. Mojo's release build
fuses some RK4 multiply-adds into FMAs (one rounding rather than two); the slow
`mu = 1e-4` recovery keeps the dynamics from being strongly chaotic, so the gap
stays small (~1.5e-12 over 50,000 steps, ~2e-8 over 2,000,000) with identical
spike counts. Mojo is validated on that bound, not whole-trace equality.

> Aligning the cube to `v*v*v` (from the historical `v**3`) made the Python
> reference bit-identical to the engine's existing `v.powi(3)`; the `v**3` path's
> libm-pow `OverflowError` was replaced by the finite guard (exact multiplication
> overflows to inf instead of raising), with the same reject-without-mutation
> contract.

### Measured backends

Reproduce with `python benchmarks/bench_fitzhugh_rinzel_simulate.py --json
benchmarks/results/bench_fitzhugh_rinzel_simulate.json`. Workload: 2,000,000 RK4
steps, default parameters, current = 0.5, median of 5 repeats. **Non-isolated**
(loaded workstation, Python 3.12 / NumPy 2.3) — functional/regression evidence,
not isolated-core release numbers.

| backend | median (ms) | speedup vs NumPy | parity Δ vs NumPy |
|---|---:|---:|---:|
| python (NumPy) | 1905.59 | 1.00× | 0 |
| mojo | 87.96 | 21.66× | 2.19e-08 (slow-growing FMA band) |
| go | 95.19 | 20.02× | 0 (bit-exact) |
| julia | 101.13 | 18.84× | 0 (bit-exact) |
| rust | 101.70 | 18.74× | 0 (bit-exact) |

Mojo is fastest in raw throughput, but it is not bit-exact, so `auto` selects
Rust — the fastest backend that is both bit-exact and ships in the wheel. The
four derivative stages over three states keep the absolute per-step cost high, so
the speedups settle around 19–22×.

---

## Relationship to FitzHugh-Nagumo

The fast subsystem is the FitzHugh-Nagumo cubic oscillator with an added slow bias term:

| Model | Fast equation |
|-------|---------------|
| FitzHugh-Nagumo | `dv/dt = v - v^3/3 - w + I` |
| FitzHugh-Rinzel | `dv/dt = v - v^3/3 - w + y + I` |

The slow `y` variable acts as an endogenous current modulation. When `y` increases, the effective drive to the fast subsystem increases; when `y` decreases, the fast subsystem moves toward quiescence. This is the minimal qualitative mechanism the project uses for three-state burst-envelope dynamics.
