<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# FitzHughRinzelNeuron

**Module:** `sc_neurocore.neurons.models.fitzhugh_rinzel`
**Rust engine:** `sc_neurocore_engine::neurons::simple_spiking::FitzHughRinzelNeuron`
**Polyglot `simulate` chain (RK4):** Rust (`py_fitzhugh_rinzel_simulate`), Julia (`FitzHughRinzelAccel`), Go (`accel/go/neurons/fitzhugh_rinzel`, c-shared), Mojo (`accel/mojo/neurons/fitzhugh_rinzel.mojo`) — see *Polyglot acceleration* below
**Reference:** Rinzel, *A Formal Classification of Bursting Mechanisms in Excitable Systems* (1987), equations (3.4)–(3.6), DOI `10.1007/978-3-642-93360-8_26`.
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

Rinzel records the system as an unpublished 1976 FitzHugh–Rinzel model and gives
`I=0.3125`, `a=0.7`, `b=0.8`, `c=-0.775`, `d=1`, `delta=0.08`, and
`mu=0.0001` for the illustrated profile. The maintained `a`–`mu` defaults
preserve that profile. Fixed-step classical RK4 at `dt=0.1`, caller-supplied
current, and sampled upward `v_threshold=1` crossings are repository
specialisations; they are not attributed to the source chapter.

---

## Numerical integration contract

All maintained implementations now use the same candidate-first fourth-order Runge-Kutta update over `(v, w, y)`:

1. Validate that mutable state, scalar parameters, current, and timestep are finite.
2. Require positive `b`, `d`, `delta`, `mu`, and `dt`.
3. Compute a full RK4 candidate without mutating public state.
4. Reject non-finite derivative or candidate values before mutation.
5. Commit the candidate and report the rising-edge threshold decision without a reset.

The RK4 candidate is equivalent to applying the standard four-stage integrator
to the coupled ODE above, not to three independent scalar updates. The Python,
production Rust/PyO3, Go, Julia, and Mojo batch paths compute into private
state, validate the complete trace and final state, and only then update the
Python object. The scalar Python and Rust safety paths use the same
candidate-first rule.

Invalid runtime input fails closed:

| Surface | Invalid configuration/input | Non-finite stage or candidate | State result |
|---------|-----------------------------|-------------------------------|--------------|
| Python reference | `ValueError` | `FloatingPointError` | unchanged |
| production Rust/PyO3 | Python conversion error or `FloatingPointError` | `FloatingPointError` | unchanged |
| Julia batch | `ArgumentError`, normalised by the dispatcher | `DomainError`, normalised by the dispatcher | unchanged |
| Go/Mojo C ABI | negative sentinel, rejected by the dispatcher | negative sentinel, rejected by the dispatcher | unchanged |
| scalar Rust engine/safety API | zero event sentinel | zero event sentinel | unchanged |

The public Python dispatcher presents batch divergence uniformly as
`FloatingPointError`; low-level non-throwing adapters retain their native
sentinel contracts.

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

The dedicated `tests/test_model_fitzhugh_rinzel_fhr_*.py`,
`tests/test_fitzhugh_rinzel_backends.py`, and
`tests/test_fitzhugh_rinzel_engine_binding.py` surfaces assert the following
contracts:

| Contract | Evidence |
|----------|----------|
| ODE formula | derivative helper equals the documented three-state RHS |
| RK4 fidelity | one-step update matches an independent RK4 reference to tight tolerance |
| finite-domain safety | invalid construction, invalid current, corrupted state, overflow, and non-finite candidates fail before mutation |
| slow-variable physics | changing `mu` changes the long-horizon `y` drift while preserving finite state |
| current regimes | moderate current produces repeated threshold events; quiescent and high-drive regimes remain deterministic |
| boundedness | long integrations remain finite and inside broad model-specific envelopes |
| edge semantics | a spike is reported only on an upward threshold crossing and applies no reset beyond the continuous RK4 candidate |
| reproducibility | identical models under identical current sequences produce identical state and spike traces |
| public integration surfaces | population, network, projection, and analysis paths preserve the model contract |

The focused runtime commands are:

```text
PYTHONPATH=bridge:src .venv/bin/pytest -q $(rg --files tests | rg 'fitzhugh_rinzel' | sort)
cargo test --manifest-path engine/Cargo.toml fitzhugh_rinzel --no-default-features
tmp_bin=$(mktemp /tmp/scn-fhr-safety-XXXXXX)
trap 'rm -f "$tmp_bin"' EXIT
rustc --test src/sc_neurocore/accel/rust/safety/fitzhugh_rinzel.rs -o "$tmp_bin" && "$tmp_bin"
(cd src/sc_neurocore/accel/go/neurons/fitzhugh_rinzel && go build -buildmode=c-shared -o libfhr.so fitzhugh_rinzel.go)
mojo build --emit shared-lib -o src/sc_neurocore/accel/mojo/neurons/libfhr.so src/sc_neurocore/accel/mojo/neurons/fitzhugh_rinzel.mojo
```

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
tests/test_cosim_fitzhugh_rinzel_q1616_precision.py::TestQ1616Precision::test_fitzhugh_rinzel_q1616_parity
tests/test_reference_fitzhugh_rinzel.py
```

The committed Q16.16 RTL passes real Yosys coarse synthesis. Its paired Q8.8
catalogue formal harness passes depth-4 SymbiYosys/Z3 reset and output safety.
This establishes bounded H2 evidence; it does not claim timing closure, PPA,
target-device execution, physical silicon, or universal real-number
equivalence.

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

Reproduce with `PYTHONPATH=bridge:src .venv/bin/python benchmarks/bench_fitzhugh_rinzel_simulate.py --json
benchmarks/results/bench_fitzhugh_rinzel_simulate.json`. Workload: 2,000,000 RK4
steps, default parameters, current = 0.5, median of 5 repeats. **Non-isolated**
(loaded workstation, Python 3.12.3 / NumPy 2.2.6) — functional/regression evidence,
not isolated-core release numbers.

| backend | median (ms) | speedup vs NumPy | parity Δ vs NumPy |
|---|---:|---:|---:|
| python (NumPy) | 7513.29 | 1.00× | 0 |
| mojo | 103.10 | 72.88× | 1.55e-08 (slow-growing FMA band) |
| go | 129.67 | 57.94× | 0 (bit-exact) |
| julia | 131.30 | 57.22× | 0 (bit-exact) |
| rust | 120.71 | 62.24× | 0 (bit-exact) |

Mojo is fastest in raw throughput, but it is not bit-exact, so `auto` selects
Rust — a bit-exact backend that ships in the wheel. The JSON packet binds the
driver, model, production and safety Rust, Go, Julia, Mojo, descriptor, paired
schemas, and independent trace by SHA-256. It is local-regression evidence only:
`production_speed_claim=false` and `hardware_measurement_claimed=false`.

---

## Relationship to FitzHugh-Nagumo

The fast subsystem is the FitzHugh-Nagumo cubic oscillator with an added slow bias term:

| Model | Fast equation |
|-------|---------------|
| FitzHugh-Nagumo | `dv/dt = v - v^3/3 - w + I` |
| FitzHugh-Rinzel | `dv/dt = v - v^3/3 - w + y + I` |

The slow `y` variable acts as an endogenous current modulation. When `y` increases, the effective drive to the fast subsystem increases; when `y` decreases, the fast subsystem moves toward quiescence. This is the minimal qualitative mechanism the project uses for three-state burst-envelope dynamics.
