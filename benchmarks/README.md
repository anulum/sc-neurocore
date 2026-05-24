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

## Scripts

| Script | Description |
|--------|-------------|
| `bench_neuron_integrators.py` | Research-only cross-language RK4 neuron integrator parity and timing for Python / Rust / Julia / Go / Mojo |
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
```

## Rust Benchmarks

Criterion benchmarks are defined in `engine/benches/` and run via:

```bash
cd engine && cargo bench
```
