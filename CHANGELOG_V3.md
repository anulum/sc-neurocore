# SC-NeuroCore v3 Engine Changelog

## [3.6.0] - 2026-02-10

### Phase 12: Fused Dense Pipeline + Fast PRNG + Batch Forward
- **Fused encode+AND+popcount**: `forward_fused()` eliminates intermediate input bitstream materialization
- **Fast PRNG switch**: xoshiro256++ for dense fast-path input encoding and numpy batch encoding
- **Batched dense API**: `DenseLayer.forward_batch_numpy()` processes N samples in one FFI call
- **New diagnostics**: criterion benches for fused dense, encode+popcount, batch dense, and PRNG throughput
- **Version/test/docs update**: bumped to 3.6.0 with Phase 12 test suite and migration notes

## [3.5.0] - 2026-02-10

### Phase 11: SIMD Pipeline Acceleration
- **SIMD fused AND+popcount**: AVX-512 VPOPCNTDQ accelerated dense inner loop with AVX2 fallback
- **SIMD Bernoulli encode**: AVX-512BW/AVX2 threshold compare path for packed Bernoulli generation
- **Flat weight storage**: Contiguous `[neuron][input][word]` packed layout for cache-friendly access
- **Zero-allocation LIF batch**: Pre-allocated numpy outputs for batch LIF APIs
- **Criterion benchmarks**: Added fused-and-popcount and SIMD Bernoulli diagnostics

## [3.4.0] - 2026-02-10

### Phase 10: SIMD Pack, LIF Optimization, Rayon Guard
- **SIMD pack vectorization**: AVX-512/AVX2/portable fast packing (closes 6x Blueprint target)
- **Branchless LIF mask**: Eliminates branches in fixed-point sign extension
- **batch_lif_run_multi()**: Parallel multi-neuron batch execution via rayon
- **Rayon work threshold**: Avoids thread-pool overhead at small input counts
- **Criterion benchmarks**: Added pack_fast, pack_dispatch, lif_100k_steps

## [3.3.0] - 2026-02-10

### Phase 9: Fast Bernoulli, Fused AND+Popcount, Zero-Copy Prepacked
- **bernoulli_packed_fast**: 8x less RNG bandwidth via byte-threshold encoding
- **Fused AND+popcount**: Eliminates intermediate buffer allocation in neuron compute
- **forward_prepacked_numpy()**: True zero-copy from numpy 2D uint64 arrays
- **set_num_threads()**: Rayon thread pool configuration for tuning parallelism
- **Criterion benchmarks**: Added bernoulli_packed_fast benchmark

## [3.2.0] - 2026-02-10

### Phase 8: Benchmark CI, Single-Call Dense Forward, Parallel Encoding
- **Criterion Benchmarks**: Expanded suite with bernoulli encoding comparison and dense forward variants
- **Benchmark CI**: Automated criterion runs with artifact upload
- **DenseLayer.forward_numpy()**: Single FFI call with numpy input/output plus parallel encoding
- **Parallel batch_encode_numpy**: Rayon-parallelized probability encoding
- **Repo cleanup**: Added local `.gitignore` for generated artifacts

## [3.1.0] - 2026-02-10

### Phase 7: Dense Forward Optimization & PyPI Publishing
- **Direct Packed Bernoulli**: `bernoulli_packed()` eliminates `Vec<u8>` intermediate allocations
- **Parallel Encoding**: `DenseLayer.forward_fast()` parallelizes input encoding with per-input RNGs
- **Pre-packed Forward**: `DenseLayer.forward_prepacked()` accepts pre-encoded numpy/list inputs and skips encoding
- **batch_encode_numpy**: Returns a 2-D numpy array instead of nested Python lists
- **PyPI Publishing**: Added automated wheel upload on `v3.*` tags via Trusted Publisher workflow
- **Updated Benchmarks**: Added dense `fast` and `prepacked` benchmark variants

## [3.0.0] - 2026-02-10

### Phase 6: Performance Optimization & Stable Release
- **NumPy Zero-Copy**: `pack_bitstream_numpy()`, `popcount_numpy()`, `unpack_bitstream_numpy()` — eliminate FFI marshalling overhead
- **Batch Operations**: `batch_lif_run()`, `batch_lif_run_varying()`, `batch_encode()` — process arrays in single FFI calls
- **Verilator CI**: Co-simulation tests run automatically on Ubuntu runners
- **Updated Benchmarks**: Formal report showing true kernel performance with zero-copy interop
- **Bridge Version Fix**: `bridge/pyproject.toml` version now matches engine

## [3.0.0-rc.1] - 2026-02-10

### Phase 5: Release Candidate
- **IR Python Bridge**: Full PyO3 bindings for ScGraphBuilder, ScGraph, verify, print, parse, emit_sv
- **Co-sim Activation**: Verilator compilation + simulation when available; graceful skip preserved
- **Wheel CI**: Cross-platform wheel builds (Linux/macOS/Windows x Python 3.9-3.12)
- **Benchmark Report**: Formal v2-vs-v3 performance comparison with Blueprint section 8 targets
- **IR Demo**: Real end-to-end Python->IR->verification->SystemVerilog demo

## [3.0.0-beta.1] - 2026-02-10

### Phase 4: HDL Compilation Pipeline
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
