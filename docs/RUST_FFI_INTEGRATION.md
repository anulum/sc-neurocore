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
│  plasticity.py ──────── create_plasticity_layer           │
│         │                                                │
│  bci_studio/bci_primitives.py ── RustEligentLearner      │
│         │                                                │
│  _native/core_engine_bridge.py ────┐                     │
│  _native/learning_bridge.py facade ┤                     │
│    ├─ learning_runtime.py (ABI)    │                     │
│    ├─ learning_rust*.py (owners)   │                     │
│    ├─ learning_wgpu.py             │                     │
│    └─ learning_torch*.py           │                     │
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
│          _native/libautonomous_learning.so              │
│  34 C-FFI symbols grouped as:                           │
│  scalar rules / ELIGENT / bounded Online O(1)           │
│  batched rules / Rayon layers / checked state transport │
│  WGPU construction, stepping, seeding, and restore      │
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
| autonomous_learning (Rust) | plasticity.py, bci_primitives.py, benchmarks/online_o1_adaptation.py | typed ctypes owners |
| tinysc_riscv (Rust no_std) | — (standalone firmware) | — |
| dynamic_adaptation (Rust) | debug/stochastic_doctor.py (overlap) | — |
| hil_debugger (Go) | — (standalone server) | WebSocket |
| interconnect (Go) | — (standalone router) | UDP binary |
| formal_proofs (Lean 4) | safety_cert/ (documentation) | — |
| neuro_safe_monitor (SV) | uvm_gen/ (testbench) | — |
| proto/ | HIL telemetry/debug consumers | Protobuf schema generation (`protoc`, `prost`, `protoc-gen-go`) |

## PyO3 LGSSM boundary

The world-model forward Kalman filter uses PyO3 rather than the core-engine C
ABI. `bridge/sc_neurocore_engine/world_model.py` exposes the compiled
`py_lgssm_kalman_filter` callable from `engine/src/lgssm.rs`; the Python loader
also accepts the root extension export for wheel-layout compatibility.

Before crossing PyO3, the dispatcher normalises observations, controls, and
model arrays into finite C-contiguous `float64` buffers. The Rust result then
passes through the same `FilterResult` shape, symmetry, positive-semidefinite,
and finite-likelihood validation used by every other backend. Explicit Rust
selection fails closed if neither export is present. RTS smoothing and EM
learning do not cross this boundary.

The source-bound artifact at
`benchmarks/results/bench_predictive_model.json` hashes `engine/src/lgssm.rs`,
the registration source, the bridge wrapper, and the installed extension used
for the measured run. It also records Rust/Python parity on the same controlled
workload used by Mojo, Go, Julia, and Python.

## Rust Safety Mirror Library

The nested crate at `src/sc_neurocore/accel/rust/` contains safety and
contract mirrors for Python surfaces that are not C-FFI entry points.  It is
excluded from the root Cargo workspace so it can be tested directly without
being confused with the PyO3 engine workspace.

Run the safety mirror suite with:

```bash
cargo test --manifest-path src/sc_neurocore/accel/rust/Cargo.toml --lib --no-default-features
```

Recent mirror hardening is covered by both Rust unit tests and Python
reference-path tests:

| Mirror module | Python authority | Rust contract coverage | Python verification surface |
|---|---|---|---|
| `safety/l7_symbolic.rs` | `sc_neurocore.scpn.layers.l7_symbolic` | parameter validation, deterministic stepping, meridian/acupoint bounds, geometry metrics, bitstream emission | `tests/test_scpn_l7_symbolic_contracts.py`, `tests/test_scpn_cross_layer.py`, `tests/test_advanced_layers.py` |
| `safety/dna_mapper.rs` | `sc_neurocore.bridges.dna_mapper` | sequence constraints, nearest-neighbour thermodynamics, strand-displacement and enzymatic gate compilation, kinetics, GF(4), plate layout | `tests/test_bridges_dna_mapper.py`, `tests/test_bridges/test_dna_mapper.py` |
| `safety/predictive_model.rs` | `sc_neurocore.world_model.predictive_model` | LGSSM shape checks, positive-definite covariance checks, Cholesky solve path, Joseph-form covariance update, log-likelihood | `tests/test_world_model/test_linear_gaussian_ssm_parameters.py`, `tests/test_world_model/test_linear_gaussian_filter_result.py`, `tests/test_world_model/test_linear_gaussian_smooth_result.py`, `tests/test_world_model/test_kalman_filter.py`, `tests/test_world_model/test_rts_smoother.py`, `tests/test_world_model/test_em_learner.py`, `tests/test_world_model/test_predictive_model_backends.py` |
| `safety/analysis.rs` | `sc_neurocore.studio.analysis` | bifurcation sweeps, sensitivity ordering, nullcline contour extraction, heatmaps, STA, frequency response, fixed-point error reporting | `tests/test_studio_analysis.py` |

`autonomous_learning` is not a safety mirror. Its authority is the compiled
crate under `crates/autonomous_learning/`, reached through the maintained C ABI.
The former `accel/rust/safety/learning_bridge.rs` transcript was removed so a
second, non-dispatched implementation cannot be mistaken for runtime parity.

## Autonomous-learning boundary contract

`learning_bridge.py` is a compatibility facade only. Runtime loading,
validation, scalar ownership, Rayon layers, WGPU layers, Torch dynamics, mixed
precision, and backend selection live in focused `learning_*` modules. Public
classes retain the historical `sc_neurocore._native.learning_bridge` identity
for imports and serialized objects.

Set `SC_NEUROCORE_LIB_PATH` to select an exact library artifact. Python and the
maintained Julia bridge both honor that path; Go links the same artifact by its
parent directory through `CGO_LDFLAGS` and `LD_LIBRARY_PATH`. The loader binds
the required ABI atomically: a missing required symbol leaves the backend
unavailable instead of publishing a partially typed library.

All owning Python wrappers implement `close()` and the context-manager
protocol. Scalar and array inputs are checked for type, shape, length, finite
values, and documented domains before crossing FFI. `RustRuleLayer` restores
opaque state only through `set_rule_layer_state_mem_checked(ptr, buffer, len)`.
The Rust parser validates magic, version, rule identifier, counts, trace
lengths, finite values, truncation, and trailing bytes before swapping any
state. The legacy length-less symbol remains ABI-compatible but is not used by
the maintained Python restore path. WGPU weights are restored through the
length-aware `set_wgpu_weights` ABI rather than a warn-and-ignore placeholder.

The Python import side treats optional engine submodules as optional
accelerators. The LGSSM loader prefers `sc_neurocore_engine.world_model` and
then checks the root extension export, so either supported wheel layout works.
If neither callable is present, the Python implementation remains importable
and explicit Rust selection fails closed. Other optional engine integrations
retain their own documented submodule requirements.

The core-engine C-FFI bridge follows the same fail-closed boundary.  Python
fallbacks remain importable when `libcore_engine.so` is absent or raises during
dynamic loading.  Native LFSR dispatch rejects null handles, unexpected output
word counts, and null bitstream pointers before exposing data to NumPy, while
the pure-Python fallback preserves deterministic recovery from an invalid zero
LFSR state.

## Performance Results

| Operation | NumPy (µs) | Rust C-FFI (µs) | Speedup |
|-----------|-----------|-----------------|---------|
| popcount 64w | 9.1 | 2.8 | 3.2× |
| popcount 256w | 9.5 | 3.1 | 3.1× |
| popcount 1024w | 12.1 | 3.3 | 3.7× |
| popcount 4096w | 19.4 | 4.9 | 4.0× |
| popcount 16384w | 59.1 | 13.3 | 4.4× |
