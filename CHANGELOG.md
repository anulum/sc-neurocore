# Changelog

All notable changes to the `sc-neurocore` project will be documented in this file.

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
