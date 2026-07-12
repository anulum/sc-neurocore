# Benchmarks

Performance benchmarks for the SC-NeuroCore framework.

## Evidence boundary

Benchmark scripts are evidence generators, not install profiles. The
cross-language Julia, Go, Mojo, WGSL, and Rust comparison harnesses exist to
measure parity and wall-clock cost for selected kernels in a source checkout.
They support internal acceleration research and backend selection; they are not
part of the default user install path and are not required by `pip install
sc-neurocore`.

Published performance claims must cite committed result artefacts under
`benchmarks/results/` or a companion paper artefact. If a script probes an
optional runtime such as Julia, Go, Mojo, CUDA, MPI, or a hardware toolchain,
absence of that runtime is an environment fact, not a failure of the base
package.

## Regression evidence gate

Benchmark artefacts that back release claims are listed in
`benchmarks/benchmark_regression_gates.json`. The gate fails closed when a
required artefact is missing, a required numeric metric is absent or non-finite,
an expected boolean/string contract changes, a declared source hash is stale, or
a declared regression threshold is crossed.

The manifest itself is part of the gate. It must carry the expected schema
version, SPDX marker, unique gate IDs, unique artefact paths, and at least one
required metric or expected-value contract per gate. A malformed or weakened
manifest is a gate failure, not a bypass.

```bash
python tools/benchmark_evidence_gate.py \
  --manifest benchmarks/benchmark_regression_gates.json \
  --output benchmarks/results/benchmark_evidence_gate_report.json
```

Do not publish benchmark claims from a changed source tree until the relevant
benchmark producer has been rerun and the gate report is regenerated from the
same checkout.

## Scripts

| Script | Description |
|--------|-------------|
| `bench_quantum_annealing_modularisation.py` | Interleaved single-file-parent/modular-candidate comparison for cold import, deterministic SC-to-Ising compilation, Python solve, process wall time, and RSS; records raw samples, source hashes, affinity, and host-load limits |
| `bench_safety_certification.py` | Interleaved parent/candidate comparison for safety-package import and deterministic in-memory evidence generation; records raw samples, source hashes, affinity, host-load limits, and non-applicability of removed report-generator mirrors |
| `bench_cli_startup.py` | Interleaved parent/candidate cold-start comparison for Python CLI import, `--version` dispatch, process wall time, and maximum RSS; records source hashes, affinity, governor, and host-load limits |
| `bench_dna_mapper_import.py` | Interleaved flat-parent/modular comparison for DNA mapper import, repeated one-gate compilation, process wall time, and maximum RSS; records raw samples, source hashes, and host limitations |
| `bench_neuron_integrators.py` | Research-only cross-language RK4 neuron integrator parity and timing for Python / Rust / Julia / Go / Mojo |
| `bench_live_control_updates.py` | Local regression evidence for generated live-control update sequences, static RTL regeneration, staged overflow/underflow trap capture, and selected sticky-trap clear semantics |
| `benchmark_regression_gates.json` | Manifest of benchmark artefacts and metrics enforced by `tools/benchmark_evidence_gate.py` |
| `bench_v2_vs_v3.py` | Compare v2 (pure-Python) vs v3 (Rust engine) performance |
| `benchmark_advanced_modules.py` | Benchmark advanced module operations |
| `benchmark_sc.py` | Core stochastic computing primitives |
| `benchmark_suite.py` | Full 14-benchmark suite across 5 categories |

## Running

```bash
# Quick run
python benchmarks/benchmark_suite.py

# Full run (10x iterations) with markdown output
python benchmarks/benchmark_suite.py --full --markdown

# v2 vs v3 comparison
python benchmarks/bench_v2_vs_v3.py

# RK4 neuron integrator parity + timing
python benchmarks/bench_neuron_integrators.py

# DNA mapper refactor regression evidence
python benchmarks/bench_dna_mapper_import.py \
  --baseline-root <parent-root> \
  --candidate-root . \
  --output benchmarks/results/local_python_2026-07-12_dna_mapper_import.json

# Safety-evidence modularisation regression evidence
python benchmarks/bench_safety_certification.py \
  --baseline-root <parent-root> \
  --candidate-root . \
  --output benchmarks/results/local_python_2026-07-12_safety_certification.json

# Quantum-annealing modularisation regression evidence
python benchmarks/bench_quantum_annealing_modularisation.py \
  --baseline-root <parent-root> \
  --candidate-root . \
  --output benchmarks/results/local_python_2026-07-12_quantum_annealing_modularisation.json
```

The DNA import harness measures Python orchestration only. Its maintained Rust
safety mirror has a separate six-test contract. Earlier generated Julia and
Mojo DNA files did not parse, and the Go file contained only empty functions;
those false surfaces are not performance evidence and are no longer shipped.

The safety-evidence harness also measures Python orchestration rather than a
numerical kernel. Its committed 30-sample local run is source-hash bound and
records the non-exclusive workstation load. Median cold import, in-memory
generation, process wall, and maximum RSS changed by +16.62%, +37.31%, +6.18%,
and -1.31%, respectively. The generation median remains 0.172 ms while the
modular path performs full-field hashing and emits fail-closed reports. These
measurements are regression context, not publishable throughput claims. The
separate `SafetyMonitor` acceleration chain is outside this benchmark.

The quantum-annealing modularisation harness measures Python orchestration and
an explicitly selected Python solver. Its committed 30-sample local run is
source-hash bound and records median compile, cold import, solve, process wall,
and RSS changes of +99.65%, -25.37%, -16.67%, -23.21%, and -3.48%. Compile
remains 0.078 ms. CPU affinity was not exclusive and host load rose during the
capture, so the artefact is regression context rather than publishable
throughput or quantum-speedup evidence. The maintained Rust authority is
`engine/src/quantum.rs`; removed generated Rust safety, Go, Julia, and Mojo
files were nonfunctional mirrors, not comparison backends.

## Rust Benchmarks

Criterion benchmarks are defined in `engine/benches/` and run via:

```bash
cd engine && cargo bench
```
