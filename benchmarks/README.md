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
```

The DNA import harness measures Python orchestration only. Its maintained Rust
safety mirror has a separate six-test contract. Earlier generated Julia and
Mojo DNA files did not parse, and the Go file contained only empty functions;
those false surfaces are not performance evidence and are no longer shipped.

## Rust Benchmarks

Criterion benchmarks are defined in `engine/benches/` and run via:

```bash
cd engine && cargo bench
```
