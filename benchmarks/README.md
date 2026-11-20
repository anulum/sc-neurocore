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

A gate that maps its record's source digests (`source_hashes`) checks every
digest of a path the record holds in its tables, at any depth, against the
source at the record's commit: the map points at it, or at another place
holding the same digest of the same path.
`tests/test_tools/test_benchmark_evidence_gate.py` refuses a table digest the
gate does not check.

A record that compares a parent revision with a candidate hashes each variant's
sources inside a list, `variants.<i>.source_hashes`. A parent's digests describe
the parent revision, so they cannot be checked at the record's commit. A gate
that sets `"variant_source_hashes": "variant_revision"` checks every such digest
at its variant's own revision, in `tools/benchmark_variant_source_hashes.py`:

- a clean variant at its `git_head`, and a parent, which has none, at its
  `source_ref`;
- a variant measured on a dirty tree (`surface_dirty: true`) at the record's
  commit, the only commit that can hold the bytes it hashed. Under the
  `worktree` policy it is compared with the source on disk.

The failures are `unresolvable_variant_revision`, `stale_variant_source_hash`
and, for a dirty variant whose bytes the record's commit did not take,
`dirty_variant_not_committed`; any other mode is
`variant_source_hashes_has_unknown_mode`. The test holds the number of
list-held digests per gate, so a new one is noticed, and requires each such
gate to set the mode.

```bash
python tools/benchmark_evidence_gate.py \
  --manifest benchmarks/benchmark_regression_gates.json \
  --output benchmarks/results/benchmark_evidence_gate_report.json
```

Do not publish benchmark claims from a changed source tree until the relevant
benchmark producer has been rerun and the gate report is regenerated from the
same checkout.

The hosted CI runs the gate into its own temporary folder and fails when the
committed `benchmarks/results/benchmark_evidence_gate_report.json` is not what
the gate writes, showing the difference. A change that moves a gate id, a path,
a failure or a count commits the report again, written after the record it
gates is committed: under the `artefact_commit` policy a staged record fails the
gate, so the comparison cannot run before the commit.

## Scripts

Install the Python and Rust wheels to measure complete SSCF packets and
bounded pull streams. Use one logical CPU for both paths. Record the CPU
and activity on its SMT sibling. The sample rate labels acquisition
duration, and the benchmark measures the processing time separately.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
taskset -c 0 python benchmarks/bench_raster_stream.py compare \
  --window-size 8 --window-size 20 --window-size 64 --sample-rate-hz 20000 \
  --output framed-raster-synthetic.json
```

For the recorded input, install `h5py` and use `--sample-rate-hz 1000
--shd shd_test.h5 --recording 0`. The report retains frame and stream
digests, first-call RSS and separate encode, decode and round-trip times.
[Measured results and hardware](../docs/benchmarks/BENCHMARKS.md#framed-spike-rasters-in-python-and-rust).

Install both wheels to compare Python and Rust DSCX and ISI on the same
binary rasters. The benchmark verifies packet parity within each format
and cross-decodes before and after timing. Warm results use resident input
and rotate the execution order. Cold results include first calls in fresh
interpreters and record their wall time and peak RSS where available.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
taskset -c 0 python benchmarks/bench_delta_codec.py compare --group-size 1 \
  --group-size 8 --group-size 256 --host-profile shared --output delta-synthetic.json
```

Select an available logical CPU and record activity on its physical core.
For recorded input, install `h5py` and supply `--shd shd_test.h5` and a
`--recording` index for each selected recording. Use `--precision 5` to
measure the quantised paths and report occupancy differences from direct
binning along with the packet sizes and timings.
[MSI results and hardware profile](../docs/benchmarks/BENCHMARKS.md#inter-channel-delta-codec-in-python-and-rust).

Run the ISI comparison with the normal Python and Rust wheels installed.
The harness validates matching packets and cross-decoding before and after
timing. Each report separates encode, decode and round trip timing and
retains every measured sample. Fresh interpreter timings use the first
recording as a representative input and include separate import and process
costs. Warm throughput uses the sum of resident codec call durations over
the whole cohort. Compare runs on the same physical core and record
competing work on that core and its SMT sibling.
The [recorded comparisons](results/isi_codec_backends.json) report
Python and Rust timings with the hardware specified for every case.

```sh
python benchmarks/bench_isi_codec.py --entropy auto --output isi-synthetic.json
python benchmarks/bench_isi_codec.py --shd shd_test.h5 --entropy huffman \
  --precision 5 --output isi-shd-huffman-quantised.json
```

The recorded-input run requires `h5py` and the uncompressed
[Spiking Heidelberg Digits test split](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/).
Cramer, Stradmann, Schemmel and Zenke provide the dataset under CC BY 4.0.
The harness uses all 700 channels and bins each recording to millisecond
indices through its last event. Events that occupy the same channel and bin
coalesce in the binary raster. Quantised runs apply the requested precision
to that raster and retain their separate timing and packet sizes.

The predictive comparison exercises complete packets through
`PredictiveSpikeCodec` with both wheels installed. It times each predictor
independently and checks exact packets, statistics and cross-decoding before
and after each measured workload. Precision 1 is lossless. Larger precision
merges complete time bins before prediction. Use the same affinity and
single-thread numerical-library settings for both backends.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
taskset -c 0 python benchmarks/bench_predictive_codec.py compare \
  --predictor context --precision 1 --host-profile shared \
  --output predictive-context-synthetic.json
taskset -c 0 python benchmarks/bench_predictive_codec.py compare \
  --predictor world_model --precision 2 --host-profile shared \
  --shd shd_test.h5 --recording 0 --output predictive-world-model-shd.json
```

Choose an available logical CPU and record its physical core's sibling
activity. Each report retains the affinity, host loads, dataset and packet
digests, source and extension digests, all timing samples and the first
call in separate interpreter processes. The `world_model` Rust path retains
NumPy for probability evaluation and uses Rust for the weight update and ISI
packet. See the [paired results](../docs/benchmarks/BENCHMARKS.md#predictive-codec-python-and-rust-timing).

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
| `bench_model_theta.py` | Source-hashed Python/Rust/Julia/Go/Mojo complete-packet parity and timing for the Theta exact held-drive flow, including byte-identical aligned events, bounded circular phase, arbitrary phase/timestep Rust/PyO3, executable safety Rust, receipt/descriptor/schema/RTL/formal/Yosys/readiness custody, stable dispatcher and measured warm-call orders, final state, affinity, runtime versions, governor, and host load |
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
mojo build --target-cpu x86-64-v3 \
  -o /tmp/evo_substrate_bench \
  src/sc_neurocore/accel/mojo/kernels/evo_substrate_bench.mojo
JULIA_DEPOT_PATH="$PWD/build/julia-depot" PYTHONHASHSEED=0 PYTHONPATH=src \
  taskset -c <cpu> .venv/bin/python \
  benchmarks/bench_evo_substrate_multilang.py \
  --samples 30 --warmups 2 --mojo-binary /tmp/evo_substrate_bench \
  --output benchmarks/results/bench_evo_substrate_multilang.json
```

The Mojo binary must be built from the current benchmark source with the
pinned toolchain; its SHA-256 is recorded in the evidence. Without
`--mojo-binary`, the harness invokes `mojo run` for every sample. Both modes
include compilation outside the timed kernel loop. The schema-v2 artefacts
record host load and CPU affinity. Their timings are regression context only;
compare runs on the same hardware and runtime. Rerun on reserved isolated
cores, with governor, frequency, versions, and load evidence retained, before
publishing performance claims.

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
