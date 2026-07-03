© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AFFERO GENERAL PUBLIC LICENSE v3
Commercial Licensing: Available

# SC-NeuroCore v3 Migration Guide

## Status

Rust-engine scaffolding is in place:

- Rust engine crate in `engine/`
- Python bridge package in `bridge/sc_neurocore_engine/`
- v2-vs-v3 equivalence tests in `tests/equivalence/`

## Build (Local)

```powershell
cd 03_CODE/sc-neurocore/engine
maturin develop --release
```

## Quick Sanity Check

```powershell
python -c "import sc_neurocore_engine; print(sc_neurocore_engine.__version__); print(sc_neurocore_engine.simd_tier())"
```

## Equivalence Tests

```powershell
cd 03_CODE/sc-neurocore
$env:PYTHONPATH=\"src;bridge\"
python -m pytest tests/equivalence -v --tb=short
```

## Notes

- v2 package under `src/sc_neurocore/` remains untouched.
- v3 bridge is a drop-in import path for hot kernels and fixed-point neuron APIs.
- Encoder and LIF in v3 currently follow strict blueprint operation ordering
  (step-then-compare encoder, refractory override after threshold evaluation).

## Surrogate Gradients and Stochastic Attention (February 2026)

### Surrogate Gradients

SC-NeuroCore v3 introduces backpropagation support for stochastic
computing layers via surrogate gradients:

- `SurrogateLif` - LIF neuron with differentiable backward pass
- `DifferentiableDenseLayer` - SC layer with weight gradient computation
- Supported surrogates: FastSigmoid, SuperSpike, ArcTan, StraightThrough

### Stochastic Attention

- Rate-mode: bit-exact match with v2 (atol < 1e-12)
- SC-mode: bitstream-based matrix multiply (new v3 capability)
- Multi-head support in the integration hardening release

### Graph Neural Network

- Rate-mode: bit-exact match with v2 (atol < 1e-12)
- SC-mode: bitstream-based message passing in the integration hardening release

### Kuramoto Oscillator Solver

- High-performance phase-difference coupling
- SSGF-compatible extended solver with geometry + PGBO terms
- Pre-allocated scratch arrays, rayon parallelism
- Box-Muller noise generation with ChaCha8Rng
- Fail-closed numerical contract: Rust core and PyO3 bridge reject non-finite
  frequencies, coupling matrices, phases, SSGF matrices, field pressure,
  invalid `dt`, and negative/non-finite noise amplitudes before stepping or
  before zero-step dry-run calls; PyO3 rejects malformed SSGF matrix shapes
  before entering Rust


## SSGF Integration and Property Testing (February 2026)

### SSGF Integration

- `step_ssgf()` - Extended Kuramoto with geometry (`W`), PGBO (`h_munu`),
  and field pressure (`F*cos`) coupling terms
- Direct integration with SSGF MicroCycleEngine pipeline
- Single `sin_diff` computation shared across all coupling terms

### Property-Based Testing

- proptest coverage for all numeric modules
- Catches edge cases: overflows, NaN, extreme values

## SC Compute Graph IR and SystemVerilog Emitter (February 2026)

### SC Compute Graph IR

A Rust-native intermediate representation for SC pipelines:

- `ScGraph`: Directed acyclic graph of SC operations
- `ScGraphBuilder`: Fluent API for graph construction
- `verify()`: Static verification (SSA, type checking, acyclicity)
- `print()` / `parse()`: Stable text format with round-trip fidelity
- 11 operation types mapping to HDL primitives

### SystemVerilog Emitter

Compile IR graphs to synthesizable RTL:

- Direct instantiation of existing `hdl/` modules
- Automatic clock/reset distribution
- Constant folding for Q8.8 fixed-point parameters

### Co-Simulation Harness

Verify generated HDL against Rust golden model:

- LFSR full-cycle equivalence
- LIF neuron bit-exact comparison
- Encoder probability convergence
- Synapse AND operation verification

## IR Python Bridge and Distributable Wheels (February 2026)

### IR Python Bridge

Construct SC compute graphs from Python and compile to SystemVerilog:

```python
from sc_neurocore_engine.ir import ScGraphBuilder

b = ScGraphBuilder("my_synapse")
x = b.input("x_prob", "rate")
w = b.input("w_prob", "rate")
x_enc = b.encode(x, length=1024, seed=0xACE1)
w_enc = b.encode(w, length=1024, seed=0xBEEF)
product = b.bitwise_and(x_enc, w_enc)
count = b.popcount(product)
rate = b.div_const(count, 1024)
b.output("firing_rate", rate)

graph = b.build()
assert graph.verify() is None
sv_code = graph.emit_sv()
```

### Co-Simulation

When Verilator is installed, co-sim tests compile HDL and compare
against the Rust golden model bit-by-bit. Without Verilator,
tests skip gracefully.

### Distributable Wheels

Release automation builds pre-built `sc_neurocore_engine` wheels with
`maturin` for Python 3.10-3.14 on Linux x86_64/aarch64, macOS, and Windows.
Prefer a matching release wheel:

```bash
python -m pip install sc_neurocore_engine
```

If no wheel exists for the active platform or Python version, build from the
source checkout with a local Rust toolchain:

```bash
python -m pip install --require-hashes -r requirements/maturin.txt
cd bridge
python -m maturin develop --release
```

## NumPy Zero-Copy Interop and Batch Operations (February 2026)

### NumPy Zero-Copy Interop

For maximum performance, use the numpy-native variants:

```python
import numpy as np
import sc_neurocore_engine as v3

bits = np.random.randint(0, 2, 1_000_000, dtype=np.uint8)
packed = v3.pack_bitstream_numpy(bits)      # Zero-copy input
count = v3.popcount_numpy(packed)           # Zero-copy input
recovered = v3.unpack_bitstream_numpy(packed, len(bits))
```

The original `pack_bitstream()` and `popcount()` functions still work
with Python lists for backward compatibility.

### Batch Operations

Process entire arrays in single FFI calls:

```python
# 100K LIF steps in one call (vs 100K per-step calls)
spikes, voltages = v3.batch_lif_run(
    100_000, leak_k=20, gain_k=256, i_t=128
)

# Per-step varying current
currents = np.array([128, 200, 150], dtype=np.int16)
spikes, voltages = v3.batch_lif_run_varying(
    leak_k=20, gain_k=256, currents=currents
)
```

### CI/CD

- Verilator co-simulation runs automatically on every push (Ubuntu)
- Wheel builds on 3 OS x 4 Python versions

## Dense Forward Optimization (February 2026)

### Dense Forward Optimization

Three performance tiers for dense layer inference:

```python
import numpy as np
import sc_neurocore_engine as v3

layer = v3.DenseLayer(64, 32, 1024)
inputs = np.random.uniform(0, 1, 64)

# Original (sequential encoding)
out = layer.forward(inputs.tolist())

# Fast (parallel encoding)
out = layer.forward_fast(inputs.tolist())

# Pre-packed (skip encoding)
packed = v3.batch_encode_numpy(inputs, length=1024, seed=42)
out = layer.forward_prepacked(packed)
```

### batch_encode_numpy

Returns a 2-D numpy `uint64` array instead of nested Python lists:

```python
probs = np.array([0.3, 0.5, 0.7, 0.9])
packed = v3.batch_encode_numpy(probs, length=1024, seed=42)
# packed.shape == (4, 16)  # 4 inputs x ceil(1024/64) words
# packed.dtype == np.uint64
```

## Single-Call Dense Forward with NumPy (February 2026)

### Single-Call Dense Forward with NumPy

The recommended high-performance inference API:

```python
import numpy as np
import sc_neurocore_engine as v3

layer = v3.DenseLayer(64, 32, 1024)
inputs = np.random.uniform(0, 1, 64)

# Single FFI call: numpy in -> parallel encode -> parallel compute -> numpy out
out = layer.forward_numpy(inputs)
# out is a numpy float64 array of shape (32,)
```

### Parallel Batch Encoding

`batch_encode_numpy` now uses rayon-parallel encoding:

```python
probs = np.random.uniform(0, 1, 1000)
packed = v3.batch_encode_numpy(probs, length=1024, seed=42)
# Each probability encoded on its own thread
```

Note: `batch_encode_numpy` uses per-index seeding (`seed + index`) for
parallelism. Use `batch_encode` for sequential single-RNG seeding.

## Fast Bernoulli Encoding and Zero-Copy Prepacked Forward (February 2026)

### Fast Bernoulli Encoding

`forward_fast` and `batch_encode_numpy` now use byte-threshold Bernoulli
encoding with 8x less random number generation overhead. This provides
1/256 probability granularity, which is negligible compared to the
statistical noise of 1024-bit bitstreams.

The original `forward()` and `batch_encode()` retain f64-precision
encoding for backward compatibility.

### Zero-Copy Prepacked Forward

For maximum throughput with pre-encoded inputs:

```python
import numpy as np
import sc_neurocore_engine as v3

layer = v3.DenseLayer(64, 32, 1024)
probs = np.random.uniform(0, 1, 64)

# Encode once, forward many times (zero-copy)
packed = v3.batch_encode_numpy(probs, length=1024, seed=42)
out = layer.forward_prepacked_numpy(packed)
# out is a numpy float64 array, packed was never copied
```

### Thread Pool Tuning

Control rayon's parallel thread count:

```python
v3.set_num_threads(4)  # Use 4 threads for all parallel ops
```

Must be called before any parallel operation. Pass 0 for automatic
(number of CPU cores).

## SIMD Pack Dispatch and Parallel LIF Batch (February 2026)

### SIMD Pack Dispatch

`pack_bitstream_numpy` now uses runtime SIMD dispatch (`AVX-512BW`, `AVX2`, or portable fallback):

```python
import numpy as np
import sc_neurocore_engine as v3

bits = np.random.randint(0, 2, 1_000_000, dtype=np.uint8)
packed = v3.pack_bitstream_numpy(bits)
```

This keeps API compatibility while accelerating large pack workloads.

### Parallel Multi-Neuron LIF Batch

Run many independent neurons in parallel:

```python
import numpy as np
import sc_neurocore_engine as v3

currents = np.full(100, 128, dtype=np.int16)
spikes, voltages = v3.batch_lif_run_multi(
    100, 100_000, leak_k=20, gain_k=256, currents=currents
)
# spikes.shape == (100, 100000)
# voltages.shape == (100, 100000)
```

### Rayon Work Threshold in Dense Fast Path

`DenseLayer.forward_fast` now selects sequential input encoding for small
input counts and rayon encoding for larger inputs. This avoids thread-pool
overhead on small workloads without changing numerical outputs (per-index
RNG seeding remains identical).

## SIMD Dense Inner Loop and Bernoulli Encoding (February 2026)

### SIMD Dense Inner Loop

Dense accumulation now uses SIMD-dispatched fused AND+popcount:

- AVX-512 (`avx512vpopcntdq`) path for 8-word vector chunks
- AVX2 path for vectorized AND + scalar lane popcount
- Portable fallback for non-SIMD targets

This path is used across dense forward variants without API changes.

### SIMD Bernoulli Encode

`forward_fast` and `batch_encode_numpy` now use `bernoulli_packed_simd`:

- AVX-512BW compare path (`64 bytes -> 64-bit mask`)
- AVX2 compare path (`2 x 32-byte compares`)
- Scalar fallback for partial words and non-SIMD systems

Sampling semantics remain statistically equivalent to SIMD pack dispatch fast encoding.

### Flat Packed Weight Storage

`DenseLayer` packed weights are now stored in one contiguous buffer:

- layout: `[neuron][input][word]`
- computed with `weight_slice(neuron, input)` accessors

This removes nested `Vec<Vec<Vec<u64>>>` indirection and improves cache locality.

### Zero-Allocation LIF Batch Outputs

`batch_lif_run`, `batch_lif_run_multi`, and `batch_lif_run_varying` now:

- pre-allocate numpy output arrays
- write directly into contiguous buffers
- avoid temporary `Vec` allocations and flatten copies

Outputs and public signatures are unchanged.

## Fused Dense Forward and Fast PRNG (February 2026)

### Fused Dense Forward Kernel

`DenseLayer.forward_fast()` now routes through a fused encode+AND+popcount path:

- no intermediate `Vec<Vec<u64>>` for encoded inputs
- per-input Bernoulli words are generated and consumed immediately
- deterministic seeding is preserved (`seed + input_idx`)

### Fast PRNG for Encode Paths

Fast encode paths now use xoshiro256++ (seeded deterministically) for improved
throughput in:

- `DenseLayer.forward()`
- `DenseLayer.forward_fast()` / fused path
- `batch_encode_numpy()`

Weight packing remains ChaCha8-based to preserve existing model weight
compatibility.

### Batched Dense Forward API

Process many dense samples in one call:

```python
import numpy as np
import sc_neurocore_engine as v3

layer = v3.DenseLayer(64, 32, 1024)
inputs = np.random.uniform(0, 1, (100, 64)).astype(np.float64)
outputs = layer.forward_batch_numpy(inputs, seed=42)
# outputs.shape == (100, 32)
```

This amortizes Python↔Rust FFI overhead and enables parallel execution over
samples.
