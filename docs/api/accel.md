# Acceleration

Backend modules for high-performance SC operations.

| Module | Purpose |
|--------|---------|
| `vector_ops` | Packed uint64 bitwise AND, popcount, pack/unpack |
| `gpu_backend` | CuPy GPU dispatch (transparent NumPy fallback) |
| `jax_backend` | JAX JIT-compiled LIF step for TPU/GPU scaling |
| `jit_kernels` | Numba-accelerated inner loops |
| `mpi_driver` | MPI-based distributed simulation |

## Rust Safety Mirrors

`src/sc_neurocore/accel/rust/` is a nested Rust crate for safety and contract
mirrors of higher-level Python modules.  It is separate from the PyO3 engine:
the mirror crate is tested directly with Cargo, while the Python modules keep
their NumPy/Python path importable when optional engine submodules are absent.

Current documented mirrors:

| Mirror | Python surface | Verification |
|---|---|---|
| `safety/analysis.rs` | `studio.analysis` | Rust unit tests plus `tests/test_studio_analysis.py` |
| `safety/dna_mapper.rs` | `bridges.dna_mapper` | Rust unit tests plus 139 DNA mapper tests |
| `safety/l7_symbolic.rs` | `scpn.layers.l7_symbolic` | Rust unit tests plus L7 and cross-layer contract tests |
| `safety/predictive_model.rs` | `world_model.predictive_model` | Rust unit tests plus 77 passed predictive-model tests, with 3 optional-path skips |

Cargo command:

```bash
cargo test --manifest-path src/sc_neurocore/accel/rust/Cargo.toml --lib --no-default-features
```

## Vector Operations

::: sc_neurocore.accel.vector_ops

## GPU Backend

::: sc_neurocore.accel.gpu_backend

## JAX Backend

::: sc_neurocore.accel.jax_backend

## JIT Kernels

::: sc_neurocore.accel.jit_kernels

## MPI Driver

::: sc_neurocore.accel.mpi_driver
