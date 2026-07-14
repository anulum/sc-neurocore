# COBA-LIF fidelity evidence

This page records the source, executable parity, Python-to-Verilog
co-simulation, reference-trace, benchmark, and formal evidence used to promote
`COBALIFNeuron` to the polyglot-complete catalogue.

## Source and maintained convention

Primary source: Brette et al. (2007), Appendix 2, Benchmark 1,
*Simulation of networks of spiking neurons: a review of tools and strategies*,
[doi:10.1007/s10827-007-0038-6](https://doi.org/10.1007/s10827-007-0038-6).
The DOI ledger was verified against Crossref and the DOI resolver on
2026-07-14.

SC-NeuroCore retains the source continuous conductance-based LIF equations and
factory constants. It advances them using the repository's coupled classical
RK4 convention. Boundary conductance increments precede integration; threshold
uses the validated raw voltage candidate; reset preserves the RK4 conductance
candidates; and the refractory voltage hold continues RK4 conductance decay.

## Executable evidence matrix

| Surface | Executed contract | Result |
|---|---|---|
| Python hand model | defaults, RK4 ordering, refractory, validation, failure atomicity, batch mutation | 32 focused tests passed |
| Rust engine/PyO3 | complete 15-double state/parameter ABI plus steps/current/events | exact parity |
| Rust safety module | independent `rustc --test` execution | 5/5 passed |
| Julia | complete batch contract | exact trace/state/events |
| Go | service and C ABI builds plus complete batch contract | exact trace/state/events |
| Mojo | shared-library C ABI plus complete batch contract | max absolute difference `7.11e-15`; events exact |
| Paired schemas | TOML equals JSON; hand class equals universal runner | four states and all six events exact |
| Generated RTL | 48-bit Q24.24 four-phase RK4 datapath | all six event indices exact; bounded state error |
| Independent DOI trace | separately re-derived equations and RK4 | every feature within `1e-12` |
| SymbiYosys | depth-4 Z3 bounded reset-safety property | `PASS` |

The enrolled 400-step co-simulation protocol uses `dt=0.1`, `I=650`,
`delta_ge=0.15`, and `delta_gi=0.07`. Python, the TOML/JSON schema runners, and
the generated Q24.24 RTL all emit events at zero-based indices
`29, 103, 177, 251, 325, 399`. The asserted maximum fixed-point errors are:

| State | Maximum error |
|---|---:|
| `v` | `1.0e-5` |
| `g_e` | `5.0e-6` |
| `g_i` | `3.0e-6` |
| `refractory_time` | `2.0e-6` |

## Benchmark evidence

The artifact
`benchmarks/results/local_python_2026-06-18_coba_lif_rk4.json` records seven
warmed repetitions of 200,000 complete macro steps with a non-default state and
parameter set. It was pinned to one logical CPU, but that CPU was not exclusively
isolated. All five public lanes emitted 3,077 events, and the measured median
order was Julia, Rust, Mojo, Go, then Python. The JSON artifact retains the exact
per-run samples and medians.

Those values select Julia first for `backend="auto"` on this maintained local
artifact. They are regression evidence only: the artifact explicitly declines
a production-speed or isolated-hardware claim.

## Focused reproduction commands

From the repository root:

```bash
PYTHONPATH=src:. .venv/bin/pytest -q \
  tests/test_model_coba_lif.py \
  tests/test_coba_lif_backend_loading.py \
  tests/test_coba_lif_backends.py \
  tests/test_bench_coba_lif.py \
  tests/test_cosim_coba_lif.py \
  tests/test_reference_coba_lif.py

cargo test --manifest-path engine/Cargo.toml coba --no-default-features

(cd src/sc_neurocore/accel/go && go test ./services -run COBALIF)

PYTHONPATH=src:. .venv/bin/pytest -q tests/test_catalogue_formal.py

(cd hdl/formal/catalogue && sby -f sc_cobalifneuron.sby)

PYTHONPATH=src:. .venv/bin/python tools/benchmark_evidence_gate.py
```

The repository-wide benchmark-evidence report can include stale hashes from
unrelated models. The COBA-LIF gate is considered closed only when
`coba-lif-rk4-multibackend-local-regression` appears in `evaluated_gates` and
has no entry in `failures`; the 2026-07-14 regeneration satisfies that scoped
condition.

## Evidence surfaces

- Python model: `src/sc_neurocore/neurons/models/coba_lif.py`
- Native loader: `src/sc_neurocore/accel/coba_lif.py`
- Rust engine and PyO3: `engine/src/neurons/simple_spiking.rs`,
  `engine/src/pyo3_neurons.rs`, and `engine/src/lib.rs`
- Rust safety: `src/sc_neurocore/accel/rust/safety/coba_lif.rs`
- Julia, Go, Mojo: `src/sc_neurocore/accel/{julia,go,mojo}`
- Paired schemas: `src/sc_neurocore/neurons/model_schemas/coba_lif.{toml,json}`
- Co-simulation: `tests/test_cosim_coba_lif.py`
- Independent trace: `tests/test_reference_coba_lif.py` and
  `src/sc_neurocore/neurons/reference_trace_data/coba_lif_conductance_rk4_doi.json`
- Formal: `hdl/formal/catalogue/sc_cobalifneuron.{sby,v}` and
  `hdl/formal/catalogue/sc_cobalifneuron_formal.v`
- Benchmark and gate: `benchmarks/bench_model_coba_lif.py`, the JSON artifact,
  and `benchmarks/benchmark_regression_gates.json`

The formal job proves only its stated bounded reset-safety property. The
float64, fixed-point, and independent-reference claims come from their separate
executed tests and should not be inferred from the formal result.
