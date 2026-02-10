# SC-NeuroCore v3 Engine Changelog

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
