<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# PernarowskiNeuron

**Module:** `sc_neurocore.neurons.models.pernarowski`
**Rust engine:** `sc_neurocore_engine::neurons::simple_spiking::PernarowskiNeuron`
**Polyglot `simulate` backends:** Rust engine (PyO3), Julia `PernarowskiAccel`, Go c-shared (`accel/go/neurons/pernarowski`), Mojo FFI (`accel/mojo/neurons/pernarowski.mojo`); standalone Rust safety mirror `PernarowskiNeuron`
**Reference:** Pernarowski, M. (1994), *Fast Subsystem Bifurcations in a Slowly Varying Liénard System Exhibiting Bursting*, SIAM Journal on Applied Mathematics 54(3), 814–832, DOI `10.1137/S003613999223449X`.
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

The three-state cubic slow-fast dynamics and the autonomous bursting reference
are source-anchored. Fixed-step classical RK4, `dt=0.1`, caller-supplied `I`,
and sampled upward `v_threshold=0.5` events are explicit repository
specialisations; they are not attributed to the paper.

---

## Numerical integration contract

Python, production Rust/PyO3, Julia, Go, Mojo, and the scalar Rust safety
surface use the same candidate-first RK4 update over the coupled `(v, w, z)`
system:

1. Validate finite state, finite offsets, finite threshold, positive timescale/coupling parameters, and finite current.
2. Evaluate all four RK4 derivative stages against the same coupled ODE.
3. Reject derivative overflow, non-finite derivative output, and non-finite candidate state before mutation.
4. Commit `(v, w, z)` only after candidate validation.
5. Emit a spike only for upward threshold crossings, without artificial reset.

Invalid runtime input fails closed:

| Surface | Invalid configuration/input | Non-finite stage or candidate | State result |
|---------|-----------------------------|-------------------------------|--------------|
| Python reference | `TypeError`, `ValueError`, or `FloatingPointError` | `FloatingPointError` | unchanged |
| production Rust/PyO3 | Python conversion error or `FloatingPointError` | `FloatingPointError` | unchanged |
| Julia batch | `ArgumentError`, normalised by the dispatcher | `DomainError`, normalised by the dispatcher | unchanged |
| Go/Mojo C ABI | negative sentinel, rejected by the dispatcher | negative sentinel, rejected by the dispatcher | unchanged |
| scalar Rust engine/safety API | zero event sentinel | zero event sentinel | unchanged |

The public Python dispatcher validates the complete trace and final state before
updating the object and presents native batch divergence uniformly as
`FloatingPointError`.

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

The dedicated `tests/test_model_pernarowski_pernarowski_*.py`,
`tests/test_pernarowski_backends.py`, and
`tests/test_pernarowski_engine_binding.py` surfaces assert:

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

Focused runtime commands are:

```text
PYTHONPATH=bridge:src .venv/bin/pytest -q $(rg --files tests | rg 'pernarowski' | sort)
cargo test --manifest-path engine/Cargo.toml pernarowski --no-default-features
tmp_bin=$(mktemp /tmp/scn-pernarowski-safety-XXXXXX)
trap 'rm -f "$tmp_bin"' EXIT
rustc --test src/sc_neurocore/accel/rust/safety/pernarowski.rs -o "$tmp_bin" && "$tmp_bin"
(cd src/sc_neurocore/accel/go/neurons/pernarowski && go build -buildmode=c-shared -o libpernarowski.so pernarowski.go)
mojo build --emit shared-lib -o src/sc_neurocore/accel/mojo/neurons/libpernarowski.so src/sc_neurocore/accel/mojo/neurons/pernarowski.mojo
```

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
ULP-scale difference stays bounded on the enrolled periodic workloads and the
spike counts match. `auto` selects the wheel-shipped, bit-exact Rust backend;
the local timing order is not part of that dispatch contract.

### Measured throughput

2,000,000 RK4 steps, default bursting regime (`current=0.0`), median of 5 repeats.
Non-isolated loaded workstation (Intel i5-11600K) per
`BROADCAST_2026-06-04_benchmark_core_isolation` — functional/regression evidence,
not an isolated-core figure. Reproduce with
`PYTHONPATH=bridge:src .venv/bin/python benchmarks/bench_pernarowski_simulate.py --json benchmarks/results/bench_pernarowski_simulate.json`.

| Backend | Median (ms) | Speed-up vs Python | Whole-trace parity |
|---------|------------:|-------------------:|--------------------|
| python  | 5776.71 | 1.0× | reference |
| mojo    | 107.20 | 53.89× | 1.06×10⁻¹² (observed FMA band) |
| go      | 107.54 | 53.71× | bit-exact (0) |
| julia   | 142.78 | 40.46× | bit-exact (0) |
| rust (`auto`) | 125.30 | 46.10× | bit-exact (0) |

The JSON artefact binds the driver, model, production and safety Rust, Go,
Julia, Mojo, descriptor, paired schemas, and independent DOI-backed trace by
SHA-256. It is local regression evidence only:
`production_speed_claim=false` and `hardware_measurement_claimed=false`.

---

## Schema-to-RTL co-simulation

The bundled `pernarowski` TOML and JSON schemas mirror the maintained Python
contract: simultaneous classical RK4 over `v`, `w`, and `z` at `dt=0.1`, the
exact `v * v * v` operation order, rising-edge `v >= v_threshold` detection,
and no reset. At each enrolled 5,000-step operating point (`I=-0.1`, `0.0`,
`0.1`, and `0.2`), the hand model, schema runner, and emitted Q16.16 RTL report
exactly 17 crossings. The varied-drive schema-format test additionally requires
exact `v`/`w`/`z` state equality after every step and covers all 17 subsequent
below-threshold re-arms.

The committed `pernarowski_autonomous_bursting_doi` trace independently
re-derives the three-state RK4 recurrence and checks spike count, first-spike
step, and the final/minimum/maximum/mean of every state variable. Its provenance
is Pernarowski's 1994 *Fast Subsystem Bifurcations in a Slowly Varying Liénard
System Exhibiting Bursting*, DOI `10.1137/S003613999223449X`.

The committed Q16.16 RTL passes real Yosys coarse synthesis. The paired Q8.8
equation-compiler RTL and port-only harness pass a depth-4 SymbiYosys/Z3 bounded
check that asynchronous reset clears the spike output. Together with
co-simulation, this establishes an honest H2 boundary. It does not establish
timing closure, PPA, target-device execution, physical silicon, universal
real-number equivalence, or a formal functional-equivalence proof.

Focused evidence:

```text
tests/test_cosim_pernarowski_q1616_precision.py::TestQ1616Precision::test_pernarowski_q1616_parity
tests/test_cosim_pernarowski_q1616_precision.py::test_yosys_synthesises_committed_rtl
tests/test_reference_pernarowski.py
tests/test_catalogue_formal.py::test_catalogue_formal_inventory_matches_perfect_count
```

---

## Infrastructure pipeline

```text
PernarowskiNeuron
├── Python reference: candidate-first RK4, continuous threshold-crossing event
├── Rust engine/PyO3: fallible, failure-atomic candidate-first RK4 batch
├── Julia mirror: validated candidate-first RK4 batch
├── Go mirror: validated C-ABI candidate-first RK4 batch
├── Mojo mirror: validated C-ABI candidate-first RK4 batch
├── Rust safety mirror: candidate-first RK4 fail-closed path
├── Schema-to-RTL: Q16.16 exact spike-count co-simulation and Yosys synthesis
├── Formal catalogue: Q8.8 depth-4 reset-spike safety BMC
├── Population / Network / Monitor integration
└── Spike-count analysis integration
```
