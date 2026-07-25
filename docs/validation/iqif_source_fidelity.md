# IQIF source-to-silicon fidelity evidence

This page records the source binding, independent recurrence, five-language
parity, schema execution, Python-to-Verilog co-simulation, bounded formal job,
and controlled benchmark used to promote `IntegerQIFNeuron` to the
polyglot-complete catalogue.

## Immutable source binding

The scientific source is Wu et al. (2021), *Integer Quadratic
Integrate-and-Fire (IQIF): A Neuron Model for Digital Neuromorphic Systems*,
[doi:10.1109/AICAS51828.2021.9458572](https://doi.org/10.1109/AICAS51828.2021.9458572).
The executable source is the coauthor repository `twetto/iq-neuron`, pinned at
commit `a8752eba49dba9ba43a64be74090b91a51044b2f`:

| Source | SHA-256 |
|---|---|
| `src/iq_neuron.cpp` | `1c7fb3184a82a1fdd8a2c29a7420da62987fce953bf8ce33a5b08fea2f880b99` |
| `include/iq_neuron.h` | `4beac58c1685cb19332ec6067bfa5eb2ce3c15d9196ba6ddd6aaa2ca608f9de1` |

That implementation starts at rest, derives the piecewise branch with C++
truncation toward zero, applies a Q0.3 arithmetic shift, adds current and
zero-valued tutorial noise, emits only when the candidate is strictly greater
than `v_max`, resets to `v_reset`, and otherwise applies the lower clamp.

The historical IQIF name must not be read as permission to replace this
recurrence with `v²`. The pinned implementation is the paper's
hardware-oriented piecewise-linear approximation.

## Independent reference

`tests/test_reference_iqif.py` evaluates the literal source recurrence without
importing production model code. It serializes every row as
`<zero-based-step> <event> <post-step-v>` and pins the resulting text SHA-256:

```text
57c6916aac726033546610a2e784a22cf5b57d70657779abc7269510e9e64bf3
```

For 400 ticks at current 10 with the source defaults, the exact features are:

| Observable | Value |
|---|---:|
| events | 26 |
| first one-based event | 15 |
| inter-event period | 15 ticks |
| final state | 198 |
| minimum / maximum | 128 / 242 |
| state sum | 71,904 |
| state mean | 179.76 |

The committed artifact
`src/sc_neurocore/neurons/reference_trace_data/iqif_a8752eb_tutorial.json`
binds those features to the immutable source commit, both source hashes, the
DOI, and the standard `UniversalNeuron` runner. The hand model, TOML schema,
and JSON schema reproduce all 400 states and events with zero tolerance.

## Executable evidence matrix

| Surface | Executed contract | Result |
|---|---|---|
| Python | signed-int32 state/current, C++ branch division, Q0.3 force, strict event and lower clamp | 400 states and 26 events exact |
| Rust engine | complete eight-field state plus current and batch length through PyO3 | trace, count, and final state exact |
| Rust safety | standalone module compiled with `rustc --test` | 8/8 tests pass |
| Julia | complete integer state/parameter batch | trace, count, and final state exact |
| Go | validated service and generated C-shared ABI | trace, count, and final state exact |
| Mojo | validation pass followed by atomic output pass | trace, count, and final state exact |
| TOML/JSON DSL | `method="map"`, derived branch point, strict level event | all 400 ticks exact |
| Registered RTL | state-owning signed Q32.0 equation-compiler module | all 400 ticks exact |
| Folded RTL | combinational signed Q32.0 datapath with caller-owned state | all 400 ticks exact |
| SymbiYosys | depth-4 Z3 reset and signed-state bounded safety | `PASS` |

Boundary tests separately prove that a candidate equal to `v_max` survives,
`v_max+1` emits and hard-resets, negative candidates clamp at `v_min`, invalid
runtime mutation cannot commit state, and configured non-default contracts
travel through every native ABI without default substitution.

## Python-to-Verilog contract

The paired schemas encode the integer map as

```text
max(v_min, v + (((a * (v_rest - v)) // 8 if v < branch_point
                 else (b * (v - v_threshold)) // 8) + I))
```

The equation emitter accepts signed floor division by a positive power-of-two
integer literal and emits exactly two arithmetic `>>> 3` operations. IQIF uses
only the signed compiler path. Unsigned equation-compiler mode is not part of
this evidence or an implied capability.

Icarus Verilog executes the complete 400-tick tutorial in both production
forms. Python, registered RTL, and folded RTL have identical event/state rows,
including the strict candidate decision and reset cycle. Q32.0 is an enrolled
format for this integer protocol, not a blanket width-safety theorem for every
possible signed-int32 parameter combination.

## Controlled five-backend benchmark

`benchmarks/bench_model_iqif.py` performs one 1,000-step warm-up followed by
seven 200,000-step calls through each public dispatcher. It fails if a backend
is missing, any state differs, an event count differs, a final state differs,
the process is unpinned without acknowledgement, or the standalone Rust safety
module fails.

The committed run was pinned to logical CPU 4 without claiming exclusive
isolation:

| Backend | Median call | Median ns/step | Events | State mismatches | Final `v` |
|---|---:|---:|---:|---:|---:|
| Python | 92.358875 ms | 461.794375 | 13,333 | 0 | 165 |
| Rust | 2.437217 ms | 12.186085 | 13,333 | 0 | 165 |
| Julia | 5.550568 ms | 27.752840 | 13,333 | 0 | 165 |
| Go | 5.626361 ms | 28.131805 | 13,333 | 0 | 165 |
| Mojo | 2.261707 ms | 11.308535 | 13,333 | 0 | 165 |

Every trajectory has little-endian int64 SHA-256
`b5c84ffb7167e23d9ba3a1e4290aa93326649bd65087781e491a237ab347a4f4`.
The measured and same-host auto order is Mojo, Rust, Julia, Go, then Python.
The artifact records all raw samples, source hashes, the exact loaded
Rust/Go/Mojo binary hashes and sizes, runtime versions, affinity, governor, and
load averages. These numbers are local non-exclusive regression evidence, not
a production speed claim or hardware measurement.

## Descriptor and formal boundary

`IntegerQIFNeuron.toml` records the complete authorship, paper DOI, pinned
source, parameter/state/backend contract, exact reference digest, science S5
trajectory evidence, and silicon H1 evidence.

The catalogue emitter produces `sc_integerqifneuron.v`, its port-only formal
harness, and `sc_integerqifneuron.sby`. The depth-4 BMC proves the declared
bounded reset/spike and signed-state safety properties. It does not prove
unbounded equivalence, absence of overflow for arbitrary external RTL inputs,
synthesis timing, placement, power, or physical-device behavior.

## Scope boundary

- The soma receives the already-computed integer `I(t)`. The paper's Eq. 3
  synaptic-current generator is network-level behavior and is not hidden in
  this neuron.
- Optional source noise is fixed to zero by the enrolled tutorial. This closure
  does not claim a stochastic IQIF variant.
- The model is deterministic simulation and research RTL, not a medical,
  safety-certified, or deployed neuromorphic device.
- The benchmark gate currently also reports inherited source-hash drift in
  other model artifacts. The IQIF gate itself has zero findings.

## Reproduction

```bash
PYTHONPATH=bridge:src:. .venv/bin/python -m pytest -q \
  tests/test_model_iqif_source_dynamics.py \
  tests/test_model_iqif_validation.py \
  tests/test_model_iqif_batch.py \
  tests/test_iqif_schema_dsl.py \
  tests/test_reference_iqif.py \
  tests/test_iqif_backend_loading.py \
  tests/test_iqif_backends.py \
  tests/test_cosim_iqif.py \
  tests/test_bench_iqif_evidence_and_hashes.py \
  tests/test_bench_iqif_gates_and_rejects.py \
  tests/test_bench_iqif_probes_and_metadata.py

taskset -c 4 env PYTHONPATH=bridge:src:. .venv/bin/python \
  benchmarks/bench_model_iqif.py \
  --json benchmarks/results/local_python_2026-07-14_iqif.json

cd hdl/formal/catalogue
sby -f sc_integerqifneuron.sby
```
