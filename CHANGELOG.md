# Changelog

All notable changes to the `sc-neurocore` project will be documented in this file.

## [Unreleased]

## [3.10.0] - 2026-03-09

### MNIST-on-FPGA Demo
- **End-to-end pipeline**: `examples/mnist_fpga/demo.py` — train (sklearn digits),
  PCA 64→16, quantise Q8.8, stochastic computing inference, Verilog weight export
- Float 94.2%, Q8.8 94.2%, SC 94.0% (L=1024, sign-magnitude encoding)
- Resource estimate: 16→10 config = ~56K LUTs (fits Artix-7 100T)
- `hdl/sc_dense_matrix_layer.v` — per-neuron weight dense layer for classification

### Vivado Tooling
- `tools/vivado_impl.tcl` — non-project flow: synth → place → route (250 MHz default)
- `tools/vivado_report.py` — parse timing/utilization/power reports to JSON

### Tutorial
- `docs/tutorials/fpga_in_20_minutes.md` — 6-section FPGA deployment tutorial

### Paper
- JOSS paper updated to submission-ready state (`paper/paper.md`)
- 12 references with DOIs, MNIST demo results, Brian2 comparison, formal verification

### Documentation Overhaul
- README: benchmarks section (Rust SIMD, Brian2 comparison, Yosys synthesis)
- README: all 10 HDL modules listed with descriptions
- Zenodo DOI updated to 10.5281/zenodo.18906614
- CITATION.cff, .zenodo.json: DOI, version, author corrections
- CONTRIBUTING.md, VALIDATION.md, getting-started.md: test counts, Python version
- Yosys MODULES list updated (10 modules)

### Fixes
- Zenodo author list corrected (sole author: Miroslav Šotek)
- DOI badge in README points to latest Zenodo record

## [3.9.1] - 2026-03-08

### Benchmarks
- **20-variant Brunel translator suite**: comprehensive characterization of
  SC-NeuroCore against Brian2 across neuron models (LIF, Izhikevich, homeostatic),
  timing variants, synapse types (STDP, dot product, Sobol bitstream), layer
  architectures (JAX, recurrent, memristive), and acceleration backends
  (Numba JIT, PyTorch CUDA GTX 1060, vectorized NumPy)
- V18 Numba JIT: 9.5× speedup over per-neuron Python loop
- V19 PyTorch CUDA: 8.7× speedup on GTX 1060 6GB
- V14 Sobol bitstream: 1.04× Brian2 ratio (closest match)
- 19 translator unit tests (`test_brunel_translator.py`)
- Fix BENCHMARKS.md CPU: i5-11600K @ 3.9 GHz (AVX-512, DL Boost)
- Fix 3 delta-PSC wiring bugs: v_reset omission, R*I*dt dilution, Poisson-as-current
- Comprehensive BENCHMARKS.md with 13+ sections and measured numbers
- Rust Criterion: 31 benchmarks captured (AVX-512)
- Brian2 2.10.1 SNN comparison: Brunel balanced network head-to-head
- NeuroBench-aligned metrics: 4 configurations, up to 847 MOP/s
- v2 vs v3 PyO3 speedup: 7.3× on large dense forward (128→64)
- Advanced module benchmarks: quantum hybrid, GNN, S-Former, BCI, DVS, chaos RNG
- Yosys synthesis tooling (`tools/yosys_synth.py`, `tools/yosys_synth.tcl`)
- CuPy 14.0.1 installed for GPU VectorizedSCLayer

### Paper
- Updated JOSS paper with measured Criterion numbers (41.3 Gbit/s pack, 224 Mstep/s LIF)
- Replaced estimated FPGA claim with Yosys tooling reference

## [3.9.0] - 2026-03-06

### SCPN Layers
- **L8-L16 pure NumPy layers**: 9 new layer files completing the full 16-layer SCPN stack (`scpn/layers/l8_phase_field.py` through `l16_director.py`)
- **16-layer registry**: `LAYER_REGISTRY` dict, `create_full_stack()` now returns all 16 layers
- **Full integrated step**: `run_integrated_step()` chains L1→L16 with inter-layer coupling

### Quantum Error Correction
- **SurfaceCodeShield**: d=3 rotated surface code with X/Z stabilizers, syndrome measurement, lookup-table decoding — corrects arbitrary single-qubit errors
- Extensible to d=5 (encode/decode/syndrome paths support arbitrary odd distance)

### Benchmarks
- Fixed double-step bug in `benchmarks/snn_comparison.py` (neurons were advanced twice per timestep)
- Fixed Lava stub notes (requires Loihi 2 hardware)
- Fixed `benchmark_suite.py` output path → `benchmarks/results/`
- SNN comparison results recorded in `docs/benchmarks/BENCHMARKS.md`

### Formal Verification
- **LIF neuron**: `hdl/formal/sc_lif_neuron.sby` + `sc_lif_neuron_formal.v` — 5 properties (reset, spike-reset, refractory clamp, counter bound, spike reachability)
- **Bitstream synapse**: `hdl/formal/sc_bitstream_synapse.sby` + `sc_bitstream_synapse_formal.v` — 4 properties (AND correctness, zero propagation, full-high, input coverage)

### Testing
- 6 cross-layer coupling integration tests (`test_scpn_cross_layer.py`)
- 9 surface code QEC tests (`test_qec_surface.py`)
- Test count: 945 → 960+

### Documentation
- JOSS paper: updated test count (960), qualified LUT claim, added Brunel/NeuroBench/LFSR bib entries

## [3.8.2] - 2026-03-06

### Documentation & Adoption
- **BENCHMARKS.md**: Populated with 14 real benchmark entries (i5-11600K, NumPy 1.26.4), Rust engine Criterion numbers, comparison context, reproduction instructions
- **JOSS paper draft**: `paper/paper.md` + `paper.bib` (6 references) — statement of need, architecture, key features, QA
- **End-to-end notebook**: `notebooks/03_end_to_end_pipeline.ipynb` — 7-cell walkthrough (encode→synapse→neuron→VectorizedSCLayer→accuracy analysis)

### Testing
- **18 Hypothesis property-based tests**: Bitstream encoding roundtrip, LFSR determinism, neuron output constraints, layer shape invariants, RNG range/shape, recorder accumulation, encoder binary output
- **Test count**: 887 → 911 tests passing, 98.41% coverage

### Issues Closed
- #30: Property-based testing with Hypothesis
- #33: JOSS paper draft

## [3.8.1] - 2026-03-06

### Enterprise Hardening
- **11 CI workflows**: ci, v3-engine, v3-wheels, benchmark, docs, pre-commit, codeql, scorecard, stale, release, publish — all SHA-pinned, concurrency-grouped
- **Supply chain**: Every GitHub Action SHA-pinned (30+ refs), `pypa/gh-action-pypi-publish` pinned, dependabot groups GH Actions PRs
- **Security**: Bandit SAST in CI, dependabot security updates enabled, private vulnerability reporting enabled, CodeQL weekly schedule
- **Branch protection**: 6 required status checks (lint, test×2, spdx-guard, build, pre-commit)
- **Dockerfile**: Multi-stage build, Python 3.12, non-root user, OCI labels, healthcheck
- **Preflight gate**: `tools/preflight.py` (black + bandit + spdx-guard + pytest), `.githooks/pre-push` hook
- **Release pipeline**: `publish.yml` (PyPI OIDC trusted publisher, 12 platform wheels), `release.yml` attaches sdist to GitHub Releases
- **Repo hygiene**: `.dockerignore`, `.editorconfig`, `.gitattributes`, `CONTRIBUTORS.md`, `CODEOWNERS`, PR template, issue templates (YAML forms), dependabot commit-message prefixes
- **Labels**: 22 labels with colors (ci, security, breaking-change, hdl, performance, needs-review, pinned, roadmap, stale)
- **Settings**: Delete-branch-on-merge, wiki/projects disabled, OpenSSF Scorecard badge

### Lint Enforcement & Python Version
- **ruff check enforced in CI**: 258 unused/deprecated imports auto-fixed across 138 files
- **CI test matrix expanded**: Python 3.10, 3.11, 3.12 (dropped 3.9 — EOL, autoray/PennyLane incompatible)
- **`requires-python` bumped to `>=3.10`**: badge, classifiers, black/ruff target-version updated
- **bandit added to `[dev]` extras**: contributors can now `make lint` after `pip install -e ".[dev]"`
- **benchmark.yml permissions tightened**: `permissions: {}` at top, scoped per-job
- **SECURITY.md / SUPPORT.md**: GitHub Security Advisories link added
- **VALIDATION.md refreshed**: 1058 tests, 98% gate, ruff/bandit/spdx-guard/codeql/scorecard gates documented

---

## [3.8.0] - 2026-03-05

### Hardening & Documentation
- **Coverage gate raised to 98%**: De-omitted 6 modules (chaos/rng, analysis/explainability, physics/wolfram_hypergraph, robotics/swarm, learning/neuroevolution, spatial/*) plus bio/neuromodulation. 34 new tests, 1058 total, 98.10% coverage
- **NumPy 2.x audit**: Zero deprecated calls found — codebase fully compatible
- **Full API documentation**: 25 new mkdocstrings pages, all 44 subpackages wired into nav. Reorganized into Core / Compiler & Export / Domain Modules / Infrastructure sections
- **Stale issue automation**: `.github/workflows/stale.yml` — weekly sweep, 60+14 day lifecycle, exempt: pinned/security/roadmap
- **CI coverage gate sync**: `ci.yml` and `pyproject.toml` both enforce `fail_under = 98`

---

## [3.7.0] - 2026-02-11

### Phase 14: Polymorphic Engine -- HDC/VSA, SCPN Petri Nets, Fault-Tolerant Logic
- **HDC/VSA kernel**: `BitStreamTensor` gains `xor`, `xor_inplace`, `rotate_right`, `hamming_distance`, `bundle` methods for hyper-dimensional computing on 10,000-bit vectors
- **SIMD fused XOR+popcount**: AVX-512 VPOPCNTDQ / AVX2 / portable dispatch for hamming distance hot path
- **PyBitStreamTensor**: New `#[pyclass]` exposing full HDC algebra to Python (13 methods)
- **HDCVector**: High-level Python class with operator overloading (`*`=bind, `+`=bundle, `.similarity()`, `.permute()`)
- **PetriNetEngine**: Stochastic Colored Petri Net engine wrapping two `DenseLayer` instances for Places->Transitions->Places firing
- **Fault-tolerant logic**: Boolean logic with stochastic redundancy (1024-bit) survives 40%+ bit-flip rates
- **44 new tests**: 15 Rust integration + 20 Python HDC + 9 Python Petri Net
- **2 demos**: HDC symbolic query ("Capital of France?"), safety-critical Boolean logic with error sweep
- **Comprehensive study**: `docs/research/SC_NEUROCORE_V3.7_POLYMORPHIC_ENGINE_STUDY.md`

---

## [3.6.0] - 2026-02-10

### Phase 12: Fused Dense Pipeline + Fast PRNG + Batch Forward
- **Fused encode+AND+popcount**: `forward_fused()` eliminates intermediate input bitstream materialization
- **Fast PRNG switch**: xoshiro256++ for dense fast-path input encoding and numpy batch encoding
- **Batched dense API**: `DenseLayer.forward_batch_numpy()` processes N samples in one FFI call
- **New diagnostics**: criterion benches for fused dense, encode+popcount, batch dense, and PRNG throughput
- **Version/test/docs update**: bumped to 3.6.0 with Phase 12 test suite and migration notes

---

## [3.5.0] - 2026-02-10

### Phase 11: SIMD Pipeline Acceleration
- **SIMD fused AND+popcount**: AVX-512 VPOPCNTDQ accelerated dense inner loop with AVX2 fallback
- **SIMD Bernoulli encode**: AVX-512BW/AVX2 threshold compare path for packed Bernoulli generation
- **Flat weight storage**: Contiguous `[neuron][input][word]` packed layout for cache-friendly access
- **Zero-allocation LIF batch**: Pre-allocated numpy outputs for batch LIF APIs
- **Criterion benchmarks**: Added fused-and-popcount and SIMD Bernoulli diagnostics

---

## [3.4.0] - 2026-02-10

### Phase 10: SIMD Pack, LIF Optimization, Rayon Guard
- **SIMD pack vectorization**: AVX-512/AVX2/portable fast packing (closes 6x Blueprint target)
- **Branchless LIF mask**: Eliminates branches in fixed-point sign extension
- **batch_lif_run_multi()**: Parallel multi-neuron batch execution via rayon
- **Rayon work threshold**: Avoids thread-pool overhead at small input counts
- **Criterion benchmarks**: Added pack_fast, pack_dispatch, lif_100k_steps

---

## [3.3.0] - 2026-02-10

### Phase 9: Fast Bernoulli, Fused AND+Popcount, Zero-Copy Prepacked
- **bernoulli_packed_fast**: 8x less RNG bandwidth via byte-threshold encoding
- **Fused AND+popcount**: Eliminates intermediate buffer allocation in neuron compute
- **forward_prepacked_numpy()**: True zero-copy from numpy 2D uint64 arrays
- **set_num_threads()**: Rayon thread pool configuration for tuning parallelism
- **Criterion benchmarks**: Added bernoulli_packed_fast benchmark

---

## [3.2.0] - 2026-02-10

### Phase 8: Benchmark CI, Single-Call Dense Forward, Parallel Encoding
- **Criterion Benchmarks**: Expanded suite with bernoulli encoding comparison and dense forward variants
- **Benchmark CI**: Automated criterion runs with artifact upload
- **DenseLayer.forward_numpy()**: Single FFI call with numpy input/output plus parallel encoding
- **Parallel batch_encode_numpy**: Rayon-parallelized probability encoding
- **Repo cleanup**: Added local `.gitignore` for generated artifacts

---

## [3.1.0] - 2026-02-10

### Phase 7: Dense Forward Optimization & PyPI Publishing
- **Direct Packed Bernoulli**: `bernoulli_packed()` eliminates `Vec<u8>` intermediate allocations
- **Parallel Encoding**: `DenseLayer.forward_fast()` parallelizes input encoding with per-input RNGs
- **Pre-packed Forward**: `DenseLayer.forward_prepacked()` accepts pre-encoded numpy/list inputs and skips encoding
- **batch_encode_numpy**: Returns a 2-D numpy array instead of nested Python lists
- **PyPI Publishing**: Added automated wheel upload on `v3.*` tags via Trusted Publisher workflow
- **Updated Benchmarks**: Added dense `fast` and `prepacked` benchmark variants

---

## [3.0.0] - 2026-02-10

### Phase 6: Performance Optimization & Stable Release
- **NumPy Zero-Copy**: `pack_bitstream_numpy()`, `popcount_numpy()`, `unpack_bitstream_numpy()` — eliminate FFI marshalling overhead
- **Batch Operations**: `batch_lif_run()`, `batch_lif_run_varying()`, `batch_encode()` — process arrays in single FFI calls
- **Verilator CI**: Co-simulation tests run automatically on Ubuntu runners
- **Updated Benchmarks**: Formal report showing true kernel performance with zero-copy interop
- **Bridge Version Fix**: `bridge/pyproject.toml` version now matches engine

### Phase 5: Release Candidate (3.0.0-rc.1)
- **IR Python Bridge**: Full PyO3 bindings for ScGraphBuilder, ScGraph, verify, print, parse, emit_sv
- **Co-sim Activation**: Verilator compilation + simulation when available; graceful skip preserved
- **Wheel CI**: Cross-platform wheel builds (Linux/macOS/Windows x Python 3.9-3.12)
- **Benchmark Report**: Formal v2-vs-v3 performance comparison with Blueprint section 8 targets
- **IR Demo**: Real end-to-end Python->IR->verification->SystemVerilog demo

### Phase 4: HDL Compilation Pipeline (3.0.0-beta.1)
- **SC IR**: Rust-native intermediate representation with 11 op types
- **SV Emitter**: Compile IR graphs to synthesizable SystemVerilog
- **Co-sim**: Verilator-based verification against Rust golden model
- **CI**: Expanded test coverage to include all Phase 2-4 Python tests

### Phase 3: Integration & Hardening
- SSGF-compatible Kuramoto solver (`step_ssgf`, `run_ssgf`)
- Property-based testing with proptest (12 property tests)
- Multi-head attention (`forward_multihead`)
- SC-mode GNN (`forward_sc`)
- End-to-end training demo
- Comprehensive rustdoc

### Phase 2: Differentiation & Acceleration
- Surrogate gradient LIF (FastSigmoid, SuperSpike, ArcTan)
- DifferentiableDenseLayer for backpropagation
- Stochastic attention (rate + SC mode)
- Graph neural network layer
- Kuramoto oscillator solver
- Criterion benchmarks + v2/v3 comparison

### Phase 1: Foundation
- Rust engine with PyO3 bindings
- Bit-exact LFSR, LIF neuron, dense layer
- SIMD dispatch (AVX-512, AVX2, NEON, portable)
- Python bridge with v2-compatible API
- Equivalence test suite

---

## [2.2.0] - 2026-02-09

### Added
- **Module Discoverability**: Populated 36 stub `__init__.py` files with proper
  `__all__` exports and lazy imports. Every package now supports
  `from sc_neurocore.X import Y` without touching internals.
- **MkDocs API Documentation**: Added `mkdocs.yml` with mkdocstrings plugin,
  `docs/index.md`, `docs/getting-started.md`, `docs/architecture.md`, and 17
  API reference stubs in `docs/api/`.
- **Examples Directory**: 6 runnable example scripts demonstrating bitstream
  encoding, neuron layers, vectorized inference, SCPN stack, HDL generation,
  and ensemble consensus (`examples/01`–`06`).
- **Module Docstrings**: Added module-level docstrings to `pipeline/ingestion.py`,
  `pipeline/training.py`, `utils/model_bridge.py`, `ensembles/orchestrator.py`.

### Changed
- **Print → Logging**: Converted 60+ `print()` calls across 25 source modules
  to structured `logging` with `getLogger(__name__)` and `%`-style formatting.
  Dashboard and drivers intentionally excluded (stdout by design).
- **CI Coverage Threshold**: Raised `--cov-fail-under` from 50 to 97 in
  `.github/workflows/ci.yml` to match actual coverage.
- Version bump: 2.1.0 → 2.2.0.

### Fixed
- **Unused Imports**: Removed dead imports from 7 files (`bio/uploading.py`,
  `core/replication.py`, `core/immortality.py`, `export/onnx_exporter.py`,
  `dashboard/text_dashboard.py`, `hdl_gen/verilog_generator.py`, `viz/web_viz.py`).
- **Input Validation**: `VectorizedSCLayer.forward()` now raises `ValueError`
  on wrong-shape input instead of silently producing garbage.
- **File I/O Error Handling**: `onnx_exporter.py`, `immortality.py`,
  `verilog_generator.py`, and `replication.py` now catch `OSError` on
  file operations and log meaningful messages.

### Security
- **Pickle Allowlist**: Replaced wildcard `'numpy.core.numeric': {'*'}` with
  explicit `{'_frombuffer', 'scalar'}` in `core/immortality.py`.
- **Path Traversal Prevention**: `core/replication.py` now validates that the
  destination directory is within or below the working directory via
  `os.path.realpath()` + `os.path.relpath()`.

---

## [2.1.0] - 2026-02-08

### Fixed (Critical)
- **HDL Bitstream Encoder Seed Decorrelation**: All parallel encoders shared
  hardcoded seed `0xACE1`, producing correlated bitstreams and breaking SC
  multiplication (`P(x AND x) = P(x)` instead of `P(x)*P(w)`). Added per-instance
  `SEED_INIT` parameter with prime-stride offsets (input: `0xACE1 + i*7`,
  weight: `0xBEEF + i*13`).
- **HDL Missing Port Connections**: `noise_in` and `v_out` were floating on
  LIF neuron instances in `sc_dense_layer_core.v`. Connected via wire buses.
- **HDL Duplicate Port**: Removed duplicate `.stream_len` in `sc_neurocore_top.v`.
- **Fixed-Point Overflow**: `FixedPointLIFNeuron` now applies `_mask()` for
  proper two's complement overflow wrapping on membrane potential.

### Added
- **GPU Acceleration Backend** (`accel/gpu_backend.py`):
    - CuPy/NumPy dual-path with automatic GPU detection and CPU fallback.
    - `gpu_pack_bitstream()`, `gpu_vec_and()`, `gpu_popcount()`, `gpu_vec_mac()`.
    - `VectorizedSCLayer` auto-selects GPU when CuPy is available.
- **Performance Benchmark Suite** (`scripts/benchmark_suite.py`):
    - 14 benchmarks across 5 categories (scalar, packed ops, dense layer,
      full pipeline, GPU).
    - `--full` mode (10x iterations), `--markdown` output to `BENCHMARKS.md`.
- **CI/CD Pipeline** (`.github/workflows/sc-neurocore-ci.yml`):
    - Lint (black + mypy), Test (Python 3.9/3.11/3.12 matrix, coverage >= 60%),
      Build (wheel + install verification).
- **Co-Simulation Harness**:
    - `hdl/tb_sc_lif_neuron.v`: Verilog testbench reading stimuli.txt, writing
      results_verilog.txt for bit-exact comparison.
    - `scripts/cosim_gen_and_check.py`: CLI driver with `--generate` and `--check`.
- **Bit-True Python Models**:
    - `FixedPointLFSR`: 16-bit maximal-length LFSR (period 65535).
    - `FixedPointBitstreamEncoder`: LFSR + unsigned comparator.
    - `_mask()`: Two's complement sign-extension with overflow wrap.
- **Public API Surface**: Root `__init__.py` exports 28 symbols across 7
  subpackages. All subpackage `__init__.py` files populated.
- **Tiered Module System**: 43 subpackages categorised as `core` (7),
  `research` (24+), or `contrib` (5). Install extras: `[gpu]`, `[research]`,
  `[contrib]`.
- **Behavioural Equivalence Tests**: 29 tests covering LFSR, encoder, LIF
  neuron, full pipeline, and bit-width masking.
- **GPU Backend Tests**: 17 tests covering all GPU primitives and
  VectorizedSCLayer integration.

### Changed
- Version bump: 2.0.0 -> 2.1.0.
- `pyproject.toml`: Added tool configs (pytest, black, mypy), tiered extras.
- `VectorizedSCLayer`: Refactored to use GPU backend with CPU fallback.

---

## [2.0.0] - 2026-01-12

### Added
- **Sapience & Sentience (v2.2.0)**:
    - `MetaCognitionLoop`: Computational self-awareness and self-modeling.
    - `NeuromodulatorSystem`: Dopamine/Serotonin emotional state modulation.
    - `NeuroArtGenerator`: Generative AI for internal state expression.
    - `AsimovGovernor`: Ethical constraint system (Three Laws).
    - `MindDescriptionLanguage (MDL)`: Substrate-independent soul serialization.
    - `DigitalSoul`: Persistence and reincarnation protocols.
    - `VonNeumannProbe`: Code-level self-replication.
- **Galactic Scale (v2.1.0)**:
    - `InterstellarDTN`: Long-range delay-tolerant networking.
    - `DysonPowerGrid`: Stellar-scale energy management.
    - `KardashevEstimator`: Civilization Type metrics.
    - `DarkForestAgent`: Game-theoretic survival logic.
    - `MPIDriver`: Distributed cluster-scale simulation.
    - `SNNGeneticEvolver`: Automated architecture optimization.
- **Transcendent & Omega (v2.0.5)**:
    - `HeatDeathLayer`: Entropy-survival computing.
    - `PlanckGrid`: Spacetime lattice theoretical limits.
    - `HolographicBoundary`: 3D-to-2D info mapping (AdS/CFT).
    - `EverettTreeLayer`: Many-Worlds branching solver.
    - `WolframHypergraph`: Graph-rewrite universe evolution.
    - `CategoryTheoryBridge`: Unified mathematical functors.
    - `FormalVerifier`: SMT-based safety proofs.
- **Exotic & Frontiers (v2.0.0)**:
    - `VectorizedSCLayer`: 64-bit packed JIT-accelerated core.
    - `QuantumStochasticLayer`: VQC qubit rotation bridge.
    - `StochasticTransformerBlock`: Spike-driven attention.
    - `MemristiveDenseLayer`: Hardware-aware analog simulation.
    - `StochasticCPG`: Robotic locomotion oscillators.
    - `MyceliumLayer`: Fungal network dynamics.
    - `BCIDecoder`: Neural signal (EEG) interface.
    - `DVSInputLayer`: Event Camera (AER) processing.
    - `EnergyProfiler`: 45nm Energy/CO2 estimation.
    - `WatermarkInjector`: IP protection security backdoors.

### Optimized
- `BitstreamAverager`: 6x speedup using running sum algorithm.
- `BitstreamEncoder`: Added Sobol Sequence (LDS) mode for faster convergence.

### Fixed
- Fixed f-string syntax in Verilog generator.
- Fixed dimension mismatch in Attention mechanism.
- Addressed Windows encoding issues in documentation generation.

## [1.0.0] - 2025-12-03
- Initial Release: Stochastic Neurons, Synapses, and Basic Bitstream Utilities.
