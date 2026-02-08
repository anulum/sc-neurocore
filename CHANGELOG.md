# Changelog

All notable changes to the `sc-neurocore` project will be documented in this file.

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
