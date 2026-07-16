# EscapeRate stochastic fidelity evidence

This page records the source, executable parity, seeded statistical reference,
Python-to-Verilog co-simulation, benchmark, descriptor, and formal evidence used
to promote `EscapeRateNeuron` to the polyglot-complete catalogue.

## Source and maintained conventions

Primary source: W. Gerstner (2000), *Population Dynamics of Spiking Neurons:
Fast Transients, Asynchronous States, and Locking*, Neural Computation 12:43–89,
[doi:10.1162/089976600300015899](https://doi.org/10.1162/089976600300015899).
Equations (2.13)–(2.15) give the conditional escape intensity, survival
function, and firing-time density.

SC-NeuroCore combines that exponential intensity with a passive LIF membrane.
It advances constant-current membrane dynamics by their exact RC solution,
evaluates the hazard at the candidate voltage, converts survival over one
piecewise-constant step to `p = 1 - exp(-rho*dt)`, then performs a seeded
Bernoulli comparison. Those discretisation and ordering choices are maintained
implementation conventions.

The portable random contract is a right-shift maximal-length LFSR16 with taps
0, 2, 3, and 5. One logical trial advances eight primitive states before
comparison. Zero maps to seed `0xACE1`; an interior probability maps to
`floor(p*65535)+1`; the event predicate is `sample < threshold`. Eight is
coprime with 65,535, so the decimated stream retains the full non-zero period.

## Executable evidence matrix

| Surface | Executed contract | Result |
|---|---|---|
| Python hand model | exact RC, hazard, private RNG, reset/replay, validation, batch atomicity | focused model checks pass |
| Equation/Universal DSL | private seeded state, paired-schema parity, reset and failed-step atomicity | hand/TOML/JSON events and states exact |
| Rust engine/PyO3 | complete 9-double state/parameter ABI plus seed, steps, current, trace, events | exact parity |
| Rust safety module | standalone source compiled and executed with `rustc` | 7/7 module tests pass; 4,096-step stream matches |
| Julia | complete seeded batch contract | events, voltage, and RNG exact |
| Go | service and reproducible C-shared ABI | events, voltage, and RNG exact |
| Mojo | shared-library C ABI | events/RNG exact; voltage within `2e-14` |
| Generated RTL | registered and folded 48-bit Q24.24 seeded datapaths | full-period event vector and state exact at enrolled point |
| Independent statistical artifact | exhaustive polynomial/comparator re-derivation plus five seeds | hashes, counts, final RNG, rate, and ISI statistics pass |
| Catalogue descriptor | DOI, complete state/parameter/backend/reproducibility/silicon evidence | S5/H1 terminal descriptor; perfect under declared policy |
| SymbiYosys | depth-4 Z3 bounded safety property | `PASS` |

The configured native parity protocol uses `v=-64`, `v_rest=-68`,
`v_reset=-66`, `v_threshold=-52`, `tau_m=12.5`, `rho_0=0.02`,
`delta_u=4`, `R=1.3`, `dt=0.25`, seed `0x1234`, and `I=17`. Over 4,096
steps, every runtime emits the same 29 events and finishes at RNG state 45,999.
Rust, Julia, and Go reproduce Python's voltage trace exactly; Mojo stays within
`2e-14`.

## Independent statistical reference

The artifact
`src/sc_neurocore/neurons/reference_trace_data/escape_rate_lfsr16_statistical_v1.json`
does not import the production RNG helper. Its test independently evaluates the
documented polynomial, eight-step advance, threshold quantisation, and event
digest.

At constant `V=-50`, `rho_0=0.25`, and `dt=1`, the continuous probability is
`1-exp(-0.25) = 0.22119921692859512`. Across all 65,535 non-zero LFSR states:

- the threshold is 14,497 and the event count is exactly 14,496;
- the realised event probability is `0.22119478141451132`;
- the first and last event indices are 0 and 65,530;
- the mean inter-event interval is `4.520869265263884` steps;
- the inter-event coefficient of variation is `0.8842846076062356`;
- the final RNG state returns to `0xACE1`;
- the event-byte SHA-256 is
  `6f118617f2ecb7a54c5a7ca68ee38a80a68dd15494e361c77aa228397614bfa8`.

The same artifact pins 4,096-step event hashes, counts, and final states for
seeds `1`, `42`, `0xACE1`, `0xBEEF`, and `0xFFFF`. A second full-period native
test selects `rho*dt=-log(0.75)`, where every compiled lane emits exactly
16,383 events and the observed interval mean/CV match the geometric targets 4
and `sqrt(0.75)` within the declared bounds.

## Python-to-Verilog co-simulation

The production co-simulation compiles the registered module emitted by
`UniversalNeuron.to_verilog()` and runs all 65,535 logical trials with Icarus
Verilog. The operating point lies on the Q24.24 exponential LUT grid:
`V=v_rest=v_reset=v_threshold=-50`, `rho_0=0.25`, `dt=1`, and seed
`0xACE1`.

Python and RTL produce the same complete event vector. The RTL additionally
asserts constant voltage and reports the same:

| Observable | Exact result |
|---|---:|
| events | 14,496 |
| final LFSR state | 44,257 (`0xACE1`) |
| comparator threshold | 14,497 |
| Q24.24 probability | `2^24 - round(exp(-0.25)*2^24)` |
| voltage word | `-50 * 2^24` |

Compiler tests separately elaborate both registered and folded RNG ownership,
reject invalid pipeline/unsigned configurations, and verify advance-before-
compare and 17-bit threshold emission.

## Benchmark evidence

`benchmarks/results/local_python_2026-07-14_escape_rate_lfsr16.json` records
seven warmed repetitions of 200,000 complete seeded steps through every public
batch dispatcher. The run was pinned to one logical CPU but did not claim
exclusive CPU isolation or production throughput.

All five lanes emit 1,523 events and finish at RNG state 46,746. Rust, Julia,
and Go are bit-exact to Python; Mojo's maximum voltage error is
`1.4210854715202004e-14`. The measured median order is Julia, Rust, Mojo, Go,
then Python. The artifact retains each timing sample, host/load/affinity
context, runtime versions, source hashes, and an executed Rust-safety result.

The repository-wide benchmark-evidence report contains hash-stale artifacts
from unrelated older models. EscapeRate is closed only when
`escape-rate-seeded-lfsr16-five-backend-local-regression` appears in
`evaluated_gates` and has no corresponding failure; the scoped 2026-07-14
evaluation satisfies that condition. No repository-wide pass is claimed.

## Formal boundary

`sc_escaperateneuron.sby` runs a depth-4 Z3 bounded safety check against the
generated Q24.24 module. It proves only the stated generated safety properties
over that bound. It does not prove floating-point/native equivalence, LFSR
period, statistical quality, synthesis timing, place-and-route, or hardware
equivalence; those claims rely on their separate executed evidence.

## Focused reproduction commands

From the repository root:

```bash
PYTHONPATH=src:. .venv/bin/pytest -q \
  tests/test_model_escape_rate.py \
  tests/test_equation_builder.py \
  tests/test_universal_dsl.py \
  tests/test_escape_rate_backend_loading.py \
  tests/test_escape_rate_backends.py \
  tests/test_reference_escape_rate.py \
  tests/test_cosim_escape_rate.py \
  tests/test_bench_escape_rate.py

(cd src/sc_neurocore/accel/go && go test ./services -run EscapeRate)

PYTHONPATH=src:. .venv/bin/pytest -q tests/test_catalogue_formal.py

(cd hdl/formal/catalogue && sby -f sc_escaperateneuron.sby)

taskset --cpu-list <cpu> env PYTHONPATH=src:. .venv/bin/python \
  benchmarks/bench_model_escape_rate.py \
  --json benchmarks/results/local_python_2026-07-14_escape_rate_lfsr16.json

PYTHONPATH=src:. .venv/bin/python tools/benchmark_evidence_gate.py
```

Repository policy requires focused gates for this lane; these commands are not
an instruction to substitute a local full-suite or preflight run.

## Evidence surfaces

- Model and private RNG: `src/sc_neurocore/neurons/models/escape_rate.py` and
  `src/sc_neurocore/neurons/_stochastic_threshold.py`
- Native dispatcher: `src/sc_neurocore/accel/escape_rate.py`
- Rust engine and PyO3: `engine/src/neurons/trivial/escape_rate.rs`,
  `engine/src/pyo3_neurons.rs`, and `engine/src/lib.rs`
- Julia, Go, Mojo, Rust safety: `src/sc_neurocore/accel/`
- Paired schemas: `src/sc_neurocore/neurons/model_schemas/escape_rate.{toml,json}`
- Verilog compiler: `src/sc_neurocore/compiler/verilog_compiler.py`
- Co-simulation: `tests/test_cosim_escape_rate.py`
- Statistical reference: `tests/test_reference_escape_rate.py` and the JSON artifact
- Descriptor: `src/sc_neurocore/neurons/model_descriptors/EscapeRateNeuron.toml`
- Formal: `hdl/formal/catalogue/sc_escaperateneuron.{sby,v}` and
  `hdl/formal/catalogue/sc_escaperateneuron_formal.v`
- Benchmark: `benchmarks/bench_model_escape_rate.py`, its JSON artifact, and
  `benchmarks/benchmark_regression_gates.json`

The closure proves the declared seeded discrete implementation and enrolled
statistical properties. It does not claim cryptographic randomness, external-
simulator parity, biological fit, or hardware readiness beyond H1.
