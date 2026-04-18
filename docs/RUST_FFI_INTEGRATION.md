# Rust C-FFI Integration Architecture

## Overview

SC-NeuroCore uses a **Python-orchestration / Rust-performance** pattern.
High-level logic lives in Python (`src/sc_neurocore/`); compute-intensive
hot paths are compiled as Rust shared libraries and called via zero-copy
C-FFI bridges.

## C-FFI Bridge Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Python (mainline)                       │
│                                                          │
│  accel/vector_ops.py ──── vec_popcount(), vec_scc()      │
│         │                                                │
│  meta_plasticity.py ──── RustPlasticityRule               │
│         │                                                │
│  bci_studio/bci_primitives.py ── RustEligentLearner      │
│         │                                                │
│  _native/core_engine_bridge.py ────┐                     │
│  _native/learning_bridge.py ───────┤                     │
└─────────────────────────────────┬──┘                     │
                                  │ ctypes zero-copy        │
                                  ▼                         │
┌─────────────────────────────────┴──────────────────────┐
│            _native/libcore_engine.so (406KB)            │
│  15 C-FFI symbols:                                      │
│  sc_multiply, sc_mux, sc_popcount, sc_popcount64       │
│  sc_and_packed, sc_mux_packed, sc_popcount_packed      │
│  sc_scc_packed, sc_cordiv_packed                       │
│  lfsr_create, lfsr_step, lfsr_encode, lfsr_destroy     │
│  sc_saturating_sub, bitstream_free                      │
├────────────────────────────────────────────────────────┤
│          _native/libautonomous_learning.so (430KB)      │
│  8 C-FFI symbols:                                       │
│  create_rule, step_rule, get_rule_weight, reset_rule   │
│  destroy_rule, create_learner, step_learner            │
│  destroy_learner                                        │
└────────────────────────────────────────────────────────┘
```

## Build Instructions

### Rust Crates (core_engine, autonomous_learning)

```bash
cd crates/core_engine
cargo build --release
cp target/release/libcore_engine.so ../../src/sc_neurocore/_native/

cd ../autonomous_learning
cargo build --release
cp target/release/libautonomous_learning.so ../../src/sc_neurocore/_native/
```

### Rust Crates (tinysc_riscv, dynamic_adaptation)

```bash
cd crates/tinysc_riscv
cargo test  # 83 tests

cd ../dynamic_adaptation
cargo test  # 12 tests
```

### Go Services (requires Go 1.22+)

```bash
cd services/hil_debugger
go test -v ./...

cd services/interconnect
go test -v ./...
```

## Module Mapping

| Sandbox Module | Mainline Consumer | Bridge |
|------------------|-------------------|--------|
| core_engine (Rust) | accel/vector_ops.py | ctypes zero-copy |
| autonomous_learning (Rust) | meta_plasticity.py, bci_primitives.py | ctypes RAII class |
| tinysc_riscv (Rust no_std) | — (standalone firmware) | — |
| dynamic_adaptation (Rust) | debug/stochastic_doctor.py (overlap) | — |
| hil_debugger (Go) | — (standalone server) | WebSocket |
| interconnect (Go) | — (standalone router) | UDP binary |
| formal_proofs (Lean 4) | safety_cert/ (documentation) | — |
| neuro_safe_monitor (SV) | uvm_gen/ (testbench) | — |
| proto/ | — (unused schemas) | — |

## Performance Results

| Operation | NumPy (µs) | Rust C-FFI (µs) | Speedup |
|-----------|-----------|-----------------|---------|
| popcount 64w | 9.1 | 2.8 | 3.2× |
| popcount 256w | 9.5 | 3.1 | 3.1× |
| popcount 1024w | 12.1 | 3.3 | 3.7× |
| popcount 4096w | 19.4 | 4.9 | 4.0× |
| popcount 16384w | 59.1 | 13.3 | 4.4× |
