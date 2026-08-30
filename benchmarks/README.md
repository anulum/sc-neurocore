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
| `bench_nir_graph.py` | Interleaved monolithic-parent/modular-candidate comparison for NIR import, hardware-graph lowering, FPGA compilation, process wall time, and RSS; fails on generated-byte drift and records source hashes, affinity, governor, frequency, host load, and polyglot applicability |
| `bench_bioware.py` | Interleaved monolithic-parent/modular-candidate comparison for cold import and a deterministic MEA→AER→SC→opto pipeline; fails on canonical-output drift and records all Bioware source hashes, affinity, load, RSS, and truthful Python-only applicability |
| `bench_asic_flow.py` | Interleaved monolithic-parent/modular-candidate comparison for cold import, deterministic ASIC deck generation, manifest-bearing bundle writes, process wall time, and RSS; fails on generated-byte drift and records source hashes, affinity, load, and removed-mirror applicability |
| `bench_quantum_annealing_modularisation.py` | Interleaved single-file-parent/modular-candidate comparison for cold import, deterministic SC-to-Ising compilation, Python solve, process wall time, and RSS; records raw samples, source hashes, affinity, and host-load limits |
| `bench_safety_certification.py` | Interleaved parent/candidate comparison for safety-package import and deterministic in-memory evidence generation; records raw samples, source hashes, affinity, host-load limits, and non-applicability of removed report-generator mirrors |
| `bench_cli_startup.py` | Interleaved parent/candidate cold-start comparison for Python CLI import, `--version` dispatch, process wall time, and maximum RSS; records source hashes, affinity, governor, and host-load limits |
| `bench_dna_mapper_import.py` | Interleaved flat-parent/modular comparison for DNA mapper import, repeated one-gate compilation, process wall time, and maximum RSS; records raw samples, source hashes, and host limitations |
| `bench_evo_substrate.py` | Source-bound Python evolutionary-operator timings with raw samples, summary statistics, affinity, governor, frequency, and host-load evidence |
| `bench_evo_substrate_multilang.py` | Fail-closed Rust/Julia/Go/Mojo/Python kernel parity and timing with interleaved samples, source digest, toolchain context, and explicit unavailable-backend reporting |
| `bench_connor_stevens_mojo.py` | Source-hashed Python/Rust/Mojo parity and timing for the executable Connor-Stevens Mojo lane, including exact events, bounded six-state traces, affinity, runtime versions, and host load |
| `bench_hodgkin_huxley_mojo.py` | Source-hashed Python/Rust/Mojo parity and timing for the executable Hodgkin-Huxley Mojo lane, including exact events, bounded four-state traces, affinity, runtime versions, governor, and host load |
| `bench_adex.py` | Source-hashed Python/Rust/Julia/Go/Mojo parity and timing for the maintained AdEx baseline-Euler recurrence, including exact event vectors, bounded voltage/adaptation traces, complete packet digests, final state, affinity, runtime versions, governor, and host load |
| `bench_model_expif.py` | Source-hashed Python/Rust/Julia/Go/Mojo parity and timing for the deterministic zero-noise Fourcaud-Trocmé ExpIF source profile, including complete voltage/refractory/event packets, exact events, bounded state traces, affinity, runtime versions, governor, and host load |
| `bench_model_lapicque.py` | Source-hashed Python/Rust/Julia/Go/Mojo public-dispatch parity and timing for the Lapicque exact constant-current RC flow, including exact events, bounded voltage traces, measured order, final state, affinity, runtime versions, governor, and host load |
| `bench_model_perfect_integrator.py` | Source-hashed Python/Rust/Julia/Go/Mojo public-dispatch parity and timing for the Naud-Gerstner 2012 exact held-current integral, including strict-boundary complete voltage/event packets, digests, measured order, final state, affinity, runtime versions, governor, and host load |
| `bench_model_quadratic_if.py` | Source-hashed Python/Rust/Julia/Go/Mojo public-dispatch parity and timing for the Latham 2000 normalized QIF source profile, including complete voltage/event packets, arbitrary source parameters, executable Rust-safety evidence, receipt/schema/RTL custody, measured warm-call order, final state, affinity, runtime versions, governor, and host load |
| `bench_model_theta.py` | Source-hashed Python/Rust/Julia/Go/Mojo public-dispatch parity and timing for the Theta tangent-half-angle exact constant-current flow, including exact events, bounded circular phase, executable Rust-safety evidence, stable dispatcher and measured warm-call orders, final state, affinity, runtime versions, governor, and host load |
| `bench_model_dpi_neuron.py` | Source-hashed Python/Rust/Julia/Go/Mojo public-dispatch parity and timing for the coupled Indiveri-Stefanini-Chicca (2010) DPI equations, including exact events, bounded membrane traces, all final states, executable Rust-safety evidence, stable dispatcher and measured warm-call orders, affinity, runtime versions, governor, and host load |
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

# Connor-Stevens executable Mojo-lane closure
PYTHONPATH=src taskset -c <cpu> .venv/bin/python \
  benchmarks/bench_connor_stevens_mojo.py \
  --json benchmarks/results/bench_connor_stevens_mojo.json

# Hodgkin-Huxley executable Mojo-lane closure
PYTHONPATH=src:bridge taskset -c <cpu> .venv/bin/python \
  benchmarks/bench_hodgkin_huxley_mojo.py \
  --json benchmarks/results/bench_hodgkin_huxley_mojo.json

# AdEx five-backend baseline-Euler closure
PYTHONPATH=src:bridge taskset -c <cpu> .venv/bin/python \
  benchmarks/bench_adex.py \
  --json benchmarks/results/bench_adex.json

# ExpIF five-backend Fourcaud-Trocmé source-profile closure
PYTHONPATH=src:bridge taskset -c <cpu> .venv/bin/python \
  benchmarks/bench_model_expif.py \
  --json benchmarks/results/bench_expif.json

# Lapicque five-backend exact-flow closure
PYTHONPATH=src:bridge taskset -c <cpu> .venv/bin/python \
  benchmarks/bench_model_lapicque.py \
  --json benchmarks/results/bench_lapicque.json

# Perfect Integrator five-backend source-equation closure
PYTHONPATH=src:bridge taskset -c <cpu> .venv/bin/python \
  benchmarks/bench_model_perfect_integrator.py \
  --json benchmarks/results/bench_perfect_integrator.json

# Quadratic IF five-backend exact-flow closure
PYTHONPATH=src:bridge taskset -c <cpu> .venv/bin/python \
  benchmarks/bench_model_quadratic_if.py \
  --json benchmarks/results/local_python_2026-06-16_quadratic_if_exact_flow.json

# Theta five-backend exact-flow closure
PYTHONPATH=src:bridge taskset -c <cpu> .venv/bin/python \
  benchmarks/bench_model_theta.py \
  --json benchmarks/results/local_python_2026-06-16_theta_exact_flow.json

# DPI five-backend coupled-circuit Euler closure
PYTHONPATH=src taskset -c <cpu> .venv/bin/python \
  benchmarks/bench_model_dpi_neuron.py \
  --json benchmarks/results/local_python_2026-07-13_dpi_neuron_circuit.json

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

# ASIC-flow modularisation regression evidence
PYTHONHASHSEED=0 PYTHONPATH=src python benchmarks/bench_asic_flow.py \
  --baseline-root <parent-root> \
  --baseline-ref <parent-sha> \
  --candidate-root . \
  --iterations 30 --warmups 2 \
  --output benchmarks/results/bench_asic_flow.json

# NIR hardware-graph modularisation regression evidence
PYTHONHASHSEED=0 PYTHONPATH=src taskset -c <cpu> \
  .venv/bin/python benchmarks/bench_nir_graph.py \
  --baseline-root <parent-root> \
  --baseline-ref <parent-sha> \
  --candidate-root . \
  --iterations 30 --warmups 2 \
  --output benchmarks/results/bench_nir_graph.json

# Bioware modularisation regression evidence
PYTHONHASHSEED=0 PYTHONPATH=src taskset -c <cpu> \
  .venv/bin/python benchmarks/bench_bioware.py \
  --baseline-root <parent-root> \
  --baseline-ref <parent-sha> \
  --candidate-root . \
  --iterations 30 --warmups 2 \
  --output benchmarks/results/bench_bioware.json

# Autonomous-learning Python/Rust/Torch/Go/Julia regression evidence
PYTHONHASHSEED=0 PYTHONPATH=src \
  .venv/bin/python benchmarks/bench_autonomous_learning.py \
  --baseline-root <parent-root> \
  --baseline-lib <parent-libautonomous_learning.so> \
  --baseline-ref <parent-sha> \
  --candidate-root . \
  --candidate-lib <candidate-libautonomous_learning.so> \
  --candidate-ref working-tree-learning-bridge \
  --iterations 5 --warmups 1 --steps 1024 \
  --output benchmarks/results/bench_learning_bridge.json

# Evolutionary substrate Python evidence (30 samples)
PYTHONHASHSEED=0 PYTHONPATH=src taskset -c <cpu> \
  .venv/bin/python benchmarks/bench_evo_substrate.py \
  --samples 30 --warmups 2 \
  --output benchmarks/results/bench_evo_substrate.json

# Evolutionary substrate five-language evidence (30 interleaved samples)
JULIA_DEPOT_PATH="$PWD/build/julia-depot" PYTHONHASHSEED=0 PYTHONPATH=src \
  taskset -c <cpu> .venv/bin/python \
  benchmarks/bench_evo_substrate_multilang.py \
  --samples 30 --warmups 2 \
  --output benchmarks/results/bench_evo_substrate_multilang.json
```

The committed evolutionary-substrate schema-v2 artefacts were captured on a
non-exclusive workstation without kernel-reserved isolated cores. They record
the loaded-host condition and are local regression context only. Rerun both
producers on reserved isolated cores, with affinity, governor, frequency,
versions, and load evidence retained, before publishing performance claims.

The Connor-Stevens closure artefact was captured with single-logical-CPU
affinity but without a kernel-reserved core while the workstation was loaded.
Its timings are local regression evidence only. Its scientific contract is the
exact `0/2/9` event envelope at `I=0/10/20` and the measured `2e-6` Mojo trace
bound over 100 candidate-first RK4 macro-steps.

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

The ASIC-flow harness measures Python deck and bundle orchestration without
executing external EDA. Its parent and candidate payloads are byte-identical at
SHA-256 `ae901f9b10bdc61f0997964d6143568994625bdf89080f02cd58efbc83099653`.
The 30-sample capture records median cold import, deck generation, bundle write,
process wall, and RSS changes of +9.81%, +3.07%, +22.61%, -8.33%, and -7.41%.
CPU affinity was not exclusive and the workstation was under concurrent load,
so the timings are diagnostic only. Rust, Go, Julia, and Mojo entries record
the removal of nonfunctional mirrors rather than invented speed comparisons;
the maintained execution boundary is Yosys/OpenROAD/OpenSTA/KLayout/Magic/Netgen.

The NIR hardware-graph harness measures typed parser-to-FPGA orchestration rather
than neuron-dynamics arithmetic. Its parent and candidate payloads are
byte-identical at SHA-256
`32498fa1106229a4fe064862e20b86e0f0b1f0d42f8598d1988e06e68c13ef13` and
22,860 bytes. Across 30 interleaved samples, graph-lowering, FPGA-compilation,
cold-import, process-wall, and maximum-RSS medians changed by +7.09%, +24.98%,
-23.26%, -16.98%, and -0.97%. The run used CPU affinity without a reserved
core while other jobs could be active, so these values are local regression
diagnostics only. Rust, Go, Julia, and Mojo are not applicable to this metadata
lowering surface; the model-specific cross-language dynamics paths are separate.

## Rust Benchmarks

Criterion benchmarks are defined in `engine/benches/` and run via:

```bash
cd engine && cargo bench
```
