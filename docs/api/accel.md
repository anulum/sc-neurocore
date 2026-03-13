# Acceleration

Backend modules for high-performance SC operations.

| Module | Purpose |
|--------|---------|
| `vector_ops` | Packed uint64 bitwise AND, popcount, pack/unpack |
| `gpu_backend` | CuPy GPU dispatch (transparent NumPy fallback) |
| `jax_backend` | JAX JIT-compiled LIF step for TPU/GPU scaling |
| `jit_kernels` | Numba-accelerated inner loops |
| `mpi_driver` | MPI-based distributed simulation |

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
