<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# TermanWangOscillator

**Module:** `sc_neurocore.neurons.models.terman_wang`
**Rust engine:** `sc_neurocore_engine::neurons::simple_spiking::TermanWangOscillator`
**Polyglot `simulate` backends:** Rust engine (PyO3, bit-exact), Julia `TermanWangAccel`, Go c-shared (`accel/go/neurons/terman_wang`), Mojo FFI (`accel/mojo/neurons/terman_wang.mojo`); standalone Rust safety mirror `TermanWangOscillator`
**Reference:** Terman, D. & Wang, D. L. (1995). Physica D, 81, 148-176.
DOI: `10.1016/0167-2789(94)00205-5`.
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
cargo test --manifest-path engine/Cargo.toml tw_ -- --nocapture
pytest tests/test_terman_wang_backends.py
```

Observed results: Rust engine Terman-Wang tests pass (7 tests); the cross-backend
parity suite confirms Rust bit-exactness and the Julia/Go/Mojo ULP band.

---

## Schema-to-RTL co-simulation

The paired `terman_wang.toml` and `terman_wang.json` schemas reproduce the
maintained two-state contract: simultaneous classical RK4 at `dt=0.05`, the
cubic fast nullcline, the `tanh` recovery gate, rising-edge `v >= v_peak`
detection, and no reset. An 8,000-step varied-current sequence exercises every
RK4 stage and produces 28 crossings followed by 28 re-arms. The TOML and JSON
state trajectories remain within `1e-10` of the hand model while every spike
decision agrees step by step.

The transcendental gate makes raw state bit identity non-portable across math
libraries and fixed-point look-up tables, so the declared behavioural observable
is the crossing count. The Q16.16 equation-compiler path reproduces the complete
silent/single/train regime exactly:

| Constant current | Hand model | Schema runner | Q16.16 RTL |
|-----------------:|-----------:|--------------:|-----------:|
| `-1.0` | 0 | 0 | 0 |
| `0.0` | 1 | 1 | 1 |
| `0.5` | 3 | 3 | 3 |

Each result covers 8,000 steps. The DOI-backed
`terman_wang_legion_oscillation_doi` trace independently re-derives the coupled
RK4 recurrence at `I=0.5`, including all `v`/`w` feature statistics, the first
crossing at step 29, and the three-crossing total.

The descriptor is Science S5 / Silicon H1 and is registered with the formal
catalogue. Its generated Q8.8 RTL and port-only harness have a depth-4
SymbiYosys/Z3 reset-spike safety proof. That bounded property is not presented as
Python-to-RTL behavioural equivalence; the Q16.16 three-way results above are the
behavioural parity evidence.

---

## Polyglot acceleration

A single `step` is trivial, but an N-step run is a sequential RK4 recurrence that
does not vectorise, so a compiled inner loop genuinely beats Python.
`simulate(n_steps, current, backend="auto")` dispatches across the polyglot chain
and returns `(trace, spikes)`:

```python
from sc_neurocore.neurons.models.terman_wang import TermanWangOscillator

neuron = TermanWangOscillator()
trace, spikes = neuron.simulate(20_000, current=0.5)   # auto → Rust
```

The right-hand side mixes an exact cubic (written `v*v*v` so it matches the
engine's `v.powi(3)` to the last bit) with a `tanh` gating term. On Linux the Rust
engine resolves `tanh` to the **same glibc symbol** as Python, so the Rust backend
is bit-identical to the NumPy reference. Julia, Go and Mojo use their own
libm/`tanh` (and Mojo an FMA path), so they are within a per-step ULP band; a
two-dimensional autonomous relaxation oscillator cannot be chaotic
(Poincaré-Bendixson), so that band does not amplify over millions of steps and the
spike counts match. `auto` selects Rust (the bit-exact backend, shipped in the
wheel).

### Measured throughput

2,000,000 RK4 steps, default relaxation regime (`current=0.5`), median of 5
repeats. Non-isolated loaded workstation (Intel i5-11600K) per
`BROADCAST_2026-06-04_benchmark_core_isolation` — functional/regression evidence,
not an isolated-core figure. Reproduce with
`python benchmarks/bench_terman_wang_simulate.py`.

| Backend | Median (ms) | Speed-up vs Python | Whole-trace parity |
|---------|------------:|-------------------:|--------------------|
| python  | 2085.15 | 1.0× | reference |
| julia   | 161.85 | 12.9× | 4.4×10⁻¹⁶ (own libm tanh, ~1 ULP) |
| mojo    | 203.87 | 10.2× | 4.6×10⁻¹² (FMA + own tanh) |
| go      | 213.75 | 9.8× | 4.4×10⁻¹⁶ (own tanh, ~1 ULP) |
| rust (`auto`) | 249.33 | 8.4× | bit-exact (0) |

Artefact: `benchmarks/results/bench_terman_wang_simulate.json`. The earlier
single-language step-level criterion/Python figures remain valid regression
evidence for the per-step `step` path.

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
├── Schema/RTL: paired schema formats, Q16.16 crossing parity, Q8.8 bounded formal job
├── Population / Network / Projection / Monitor integration
└── Spike-count analysis integration
```
