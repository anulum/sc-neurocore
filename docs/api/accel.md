# Acceleration

Backend modules for high-performance SC operations.

The shipped acceleration contract is the Python package plus maintained
optional Rust engine paths and explicitly documented optional backends. The
Julia, Go, Mojo, and WGSL sources in the repository form a research-only
polyglot benchmark layer: they are used for parity checks, timing studies, and
future backend evaluation, but they are not installed by default and are not
required for deployment.

Do not treat the presence of a language mirror under `src/sc_neurocore/accel/`
as a shipped runtime guarantee. A backend becomes part of the supported user
surface only when a maintained Python entrypoint, install profile, tests, and
documentation all name that path explicitly.

| Module | Purpose |
|--------|---------|
| `backend` | Runtime selector for the shipped stochastic-inference backend |
| `vector_ops` | Packed uint64 bitwise AND, popcount, pack/unpack |
| `gpu_backend` | CuPy GPU dispatch (transparent NumPy fallback) |
| `jax_backend` | JAX JIT-compiled LIF step for TPU/GPU scaling |
| `jit_kernels` | Numba-accelerated inner loops |
| `mpi_driver` | MPI-based distributed simulation |
| `mojo.runner` | Optional pixi-launched Mojo kernel-suite build/benchmark runner; scalar helpers are explicit Python fallbacks with no per-call Mojo IPC |

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

## Backend Selector

::: sc_neurocore.accel.backend

## Vector Operations

::: sc_neurocore.accel.vector_ops

## GPU Backend

::: sc_neurocore.accel.gpu_backend

## JAX Backend

::: sc_neurocore.accel.jax_backend

## JIT Kernels

::: sc_neurocore.accel.jit_kernels

## MPI Driver

`sc_neurocore.accel.mpi_driver.MPIDriver` is an optional `mpi4py` integration
for distributing stochastic-computing arrays across MPI ranks. When `mpi4py` is
unavailable, or when the communicator is absent, the driver stays in
single-rank mode (`rank == 0`, `size == 1`) and returns the caller's local
arrays unchanged. In multi-rank gathers, only root receives the concatenated
buffer; non-root ranks return an empty array that preserves the local result
dtype.

::: sc_neurocore.accel.mpi_driver
