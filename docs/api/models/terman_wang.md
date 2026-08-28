<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# TermanWangOscillator

**Module:** `sc_neurocore.neurons.models.terman_wang`
**Rust engine:** `sc_neurocore_engine::neurons::simple_spiking::TermanWangOscillator`
**Polyglot `simulate` backends:** Rust engine (PyO3), Julia `TermanWangAccel`, Go c-shared (`accel/go/neurons/terman_wang`), Mojo FFI (`accel/mojo/neurons/terman_wang.mojo`); standalone Rust safety mirror `TermanWangOscillator`
**Reference:** Terman, D. & Wang, D. L. (1995), *Global competition and local cooperation in a network of neural oscillators*, Physica D 81(1–2), 148–176, DOI `10.1016/0167-2789(94)00205-5`.
**Family:** two-state relaxation oscillator used by LEGION-style temporal segmentation networks.

---

## Equations

The maintained model evolves two continuous dimensionless state variables:

| Variable | Role |
|----------|------|
| `v` | fast excitatory state |
| `w` | slow recovery state |

The ODE contract is:

$$\frac{dv}{dt} = 3v - v^3 + 2 - w + I + \rho$$

$$\frac{dw}{dt} = \epsilon\left(\alpha(1 + \tanh(v / \beta)) - w\right)$$

The public event is an upward threshold crossing:

$$v_{new} \geq v_{peak} \land v_{old} < v_{peak}$$

`step()` does not reset after an event. The oscillator remains on its continuous
relaxation trajectory; `reset()` is an explicit caller action only.

The DOI-backed historical source verification anchors the two-state cubic and
`tanh`-gated dynamics. Fixed-step classical RK4, `dt=0.05`, caller-supplied
current, and sampled `v_peak=1.5` events are repository specialisations; the
accessible publisher record does not establish those numerical conventions.

---

## Numerical integration contract

Python, production Rust/PyO3, Julia, Go, Mojo, and the scalar Rust safety
surface use the same candidate-first RK4 update over `(v, w)`:

1. Validate finite state, finite offsets and threshold, positive `beta`, `epsilon`, and `dt`, and finite current.
2. Evaluate all four RK4 stages against the same coupled ODE.
3. Reject any non-finite derivative, stage, candidate, trace, event count, or final state before public mutation.
4. Commit `(v, w)` only after the complete batch result passes validation.
5. Emit events only for sampled upward threshold crossings, without artificial reset.

Invalid runtime input fails closed:

| Surface | Invalid configuration/input | Non-finite stage or candidate | State result |
|---------|-----------------------------|-------------------------------|--------------|
| Python reference | `TypeError`, `ValueError`, or `FloatingPointError` | `FloatingPointError` | unchanged |
| production Rust/PyO3 | Python conversion error or `FloatingPointError` | `FloatingPointError` | unchanged |
| Julia batch | `ArgumentError`, normalised by the dispatcher | `DomainError`, normalised by the dispatcher | unchanged |
| Go/Mojo C ABI | negative sentinel, rejected by the dispatcher | negative sentinel, rejected by the dispatcher | unchanged |
| scalar Rust engine/safety API | zero event sentinel or `Result::Err` | zero event sentinel or `Result::Err` | unchanged |

The Python dispatcher validates the complete native result before updating the
object and presents rejected native batches as `FloatingPointError`.

---

## Parameters

| Parameter | Default | Contract | Description |
|-----------|---------|----------|-------------|
| `v` | `-1.5` | finite | fast excitatory state |
| `w` | `-0.5` | finite | slow recovery state |
| `alpha` | `3.0` | finite | recovery-sigmoid amplitude |
| `beta` | `0.2` | finite, positive | recovery-sigmoid voltage scale |
| `epsilon` | `0.02` | finite, positive | slow-recovery timescale ratio |
| `rho` | `0.0` | finite | constant fast-state bias |
| `dt` | `0.05` | finite, positive | integration timestep |
| `v_peak` | `1.5` | finite | sampled event threshold |

The default `epsilon=0.02` separates the slow recovery and fast cubic
timescales. It does not by itself prove a particular network-level LEGION
segmentation behaviour.

---

## Behavioural evidence

The dedicated `tests/test_model_terman_wang_*.py`,
`tests/test_terman_wang_backends.py`, and
`tests/test_terman_wang_engine_binding.py` surfaces assert:

| Contract | Evidence |
|----------|----------|
| ODE formula | derivative output equals an independent cubic/`tanh` RHS |
| RK4 fidelity | one-step state matches an independent four-stage reference |
| continuous event semantics | events occur only on upward crossings, without implicit reset |
| runtime regimes | silent, single-crossing, and oscillatory operating points preserve their enrolled event counts |
| finite-domain safety | invalid construction, current, state, scales, stages, candidates, and native results fail before mutation |
| polyglot parity | Rust is host-bit-exact; Julia, Go, and Mojo satisfy bounded complete traces and exact event counts |
| public integration | population, network, projection, monitor, and spike-count analysis remain wired |

Focused runtime commands are:

```text
PYTHONPATH=bridge:src .venv/bin/pytest -q $(rg --files tests | rg 'terman_wang' | sort)
cargo test --manifest-path engine/Cargo.toml terman_wang --no-default-features
tmp_bin=$(mktemp /tmp/scn-terman-wang-safety-XXXXXX)
trap 'rm -f "$tmp_bin"' EXIT
rustc --test src/sc_neurocore/accel/rust/safety/terman_wang.rs -o "$tmp_bin" && "$tmp_bin"
(cd src/sc_neurocore/accel/go/neurons/terman_wang && go build -buildmode=c-shared -o libtermanwang.so terman_wang.go)
mojo build --emit shared-lib -o src/sc_neurocore/accel/mojo/neurons/libtermanwang.so src/sc_neurocore/accel/mojo/neurons/terman_wang.mojo
```

---

## Polyglot acceleration

An N-step run is a sequential RK4 recurrence. The public dispatcher reaches
each maintained compiled batch without changing the model equations:

```python
from sc_neurocore.neurons.models.terman_wang import TermanWangOscillator

oscillator = TermanWangOscillator()
trace, events = oscillator.simulate(20_000, current=0.5)  # auto → Rust
```

The cubic uses the same multiplication order across runtimes. Rust resolves the
same host `tanh` as Python and is bit-exact on the recorded Linux workload.
Julia, Go, and Mojo use their own math-library implementations; parity tests
therefore enforce measured complete-trace bounds plus exact events on the
enrolled regimes. The local timing order is not part of dispatch selection.

### Measured throughput

2,000,000 RK4 steps at `current=0.5`, median of 5 repeats. This is a
non-isolated loaded-workstation run on the local Intel i5-11600K and is only
functional/regression evidence. Reproduce it with:

```text
PYTHONPATH=src .venv/bin/python benchmarks/bench_terman_wang_simulate.py --json benchmarks/results/bench_terman_wang_simulate.json
```

| Backend | Median (ms) | Speed-up vs Python | Whole-trace parity |
|---------|------------:|-------------------:|--------------------|
| python | 7369.89 | 1.00× | reference |
| mojo | 274.82 | 26.82× | 4.92×10⁻¹² |
| go | 287.38 | 25.65× | 4.44×10⁻¹⁶ |
| rust (`auto`) | 289.85 | 25.43× | bit-exact (0) |
| julia | 298.45 | 24.69× | 4.44×10⁻¹⁶ |

The JSON artefact binds the driver, model, production and safety Rust, Go,
Julia, Mojo, descriptor, paired schemas, and independent DOI-backed trace by
SHA-256. It explicitly records `production_speed_claim=false` and
`hardware_measurement_claimed=false`.

---

## Schema-to-RTL co-simulation

The paired `terman_wang` TOML and JSON schemas mirror simultaneous classical
RK4 at `dt=0.05`, the cubic fast nullcline, `tanh` recovery gate, upward
`v >= v_peak` observation, and no reset. Over 8,000 steps, the hand model,
schema runner, and Q16.16 RTL preserve exactly 0, 1, and 3 events at constant
currents `-1.0`, `0.0`, and `0.5` respectively.

The committed `terman_wang_legion_oscillation_doi` trace independently
re-derives the two-state recurrence at `I=0.5` and checks state statistics,
first event at step 29, and the three-event total.

The committed RTL passes real Yosys coarse synthesis. The paired Q8.8 RTL and
port-only harness pass a depth-4 SymbiYosys/Z3 reset/output safety check. This
is the declared H2 boundary. It does not establish timing closure, PPA,
target-device execution, physical silicon, universal real-number equivalence,
or formal functional equivalence.

Focused evidence:

```text
tests/test_cosim_terman_wang_q1616_precision.py::TestQ1616Precision::test_terman_wang_q1616_parity
tests/test_cosim_terman_wang_q1616_precision.py::test_yosys_synthesises_committed_rtl
tests/test_reference_terman_wang.py
hdl/formal/catalogue/sc_terman_wang.sby
```

---

## Infrastructure pipeline

```text
TermanWangOscillator
├── Python reference: candidate-first RK4, continuous threshold-crossing event
├── Rust engine/PyO3: fallible, failure-atomic candidate-first RK4 batch
├── Julia mirror: validated candidate-first RK4 batch
├── Go mirror: validated C-ABI candidate-first RK4 batch
├── Mojo mirror: validated C-ABI candidate-first RK4 batch
├── Rust safety mirror: candidate-first RK4 fail-closed path
├── Schema-to-RTL: Q16.16 event-count co-simulation and Yosys synthesis
├── Formal catalogue: Q8.8 depth-4 reset/output safety BMC
├── Population / Network / Projection / Monitor integration
└── Spike-count analysis integration
```
