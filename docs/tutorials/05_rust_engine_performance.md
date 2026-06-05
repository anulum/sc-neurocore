<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Rust Engine and Performance Optimization

SC-NeuroCore ships a Rust backend (`sc_neurocore_engine`) with SIMD-accelerated
popcount kernels and Rayon parallelism across neurons and inputs.

## 1. Installation and Verification

Requires Rust 1.75+ and maturin 1.x.

```bash
cd engine && maturin develop --release
sc-neurocore info
```

```
sc-neurocore 3.15.27
Python 3.12.x
Rust engine: 3.15.27 (avx2)
NumPy: 2.x.x
```

`--release` enables fat LTO + single codegen unit (`engine/Cargo.toml`
`[profile.release]`). Debug builds are 10-50x slower.

## 2. SIMD Tiers

`simd_tier()` probes CPU features at runtime and returns the highest tier:

| Tier | Accelerated Operations |
|------|----------------------|
| `avx512-vpopcntdq` | 512-bit pack, native VPOPCNTDQ, fused AND+popcount, f64 FMA |
| `avx512bw` | 512-bit pack, lookup-table popcount, f64 ops |
| `avx512f` | f64 dot/sum/max/scale only; popcount falls to AVX2 path |
| `avx2` | 256-bit Harley-Seal popcount, FMA f64 dot |
| `popcnt` | hardware `popcnt` per u64, scalar everything else |
| `neon` | AArch64 128-bit pack/popcount/f64 |
| `portable` | Rust `count_ones()` intrinsic, scalar loops |

```python
import sc_neurocore_engine as engine

engine.simd_tier()          # "avx2"
engine.set_num_threads(4)   # pin rayon pool; 0 = all cores (call before first parallel op)
```

## 3. Forward-Pass Hierarchy

The Rust `VectorizedSCLayer` exposes six forward methods. All produce
identical stochastic dot-product results; they differ in FFI overhead and
parallelism strategy.

```python
from sc_neurocore_engine.layers import VectorizedSCLayer
import sc_neurocore_engine as engine
import numpy as np

layer = VectorizedSCLayer(n_inputs=64, n_neurons=32, length=2048)
x = np.random.uniform(0, 1, 64)
```

**`forward(x.tolist())`** -- sequential encode, parallel accumulate (Rayon
when `n_neurons >= 8`). Two list copies across FFI.

**`forward_fast(x.tolist(), seed=42)`** -- parallel encode (Rayon when
`n_inputs >= 128`) + parallel accumulate. Best list-input single-sample path.

**`forward_numpy(x, seed=42)`** -- zero-copy `float64` input via
`PyReadonlyArray1`. Preferred for single samples from NumPy.

**`forward_prepacked(packed)`** / **`forward_prepacked_numpy(packed)`** --
skip encoding entirely. Pre-encode once with `batch_encode_numpy`, reuse
across layers:

```python
packed = engine.batch_encode_numpy(x, length=2048, seed=42)
out = layer.forward_prepacked_numpy(packed)  # zero-copy variant
```

**`forward_batch_numpy(batch, seed=42)`** -- `(N, n_inputs)` float64 array,
single FFI call, Rayon across samples and neurons:

```python
batch = np.random.uniform(0, 1, (100, 64))
out = layer.forward_batch_numpy(batch, seed=42)  # (100, 32)
```

**Selection guide:**

| Scenario | Method |
|----------|--------|
| Single sample, list input | `forward_fast` |
| Single sample, NumPy input | `forward_numpy` |
| Same input, multiple layers | `forward_prepacked_numpy` |
| Batch of N samples | `forward_batch_numpy` |

## 4. Python Fallback and GPU Path

Without the Rust engine, `sc_neurocore.layers.VectorizedSCLayer` uses packed
NumPy uint64 SWAR popcount (10-100x slower than Rust).

With CuPy installed, set `use_gpu=True` for GPU offload via `gpu_vec_mac`
(broadcast AND + SWAR popcount on CUDA). Effective for large layers (256+
neurons, 4096+ bit streams); for small networks Rust SIMD beats GPU round-trip.

```python
from sc_neurocore import VectorizedSCLayer
from sc_neurocore.accel import to_device, to_host, HAS_CUPY

layer = VectorizedSCLayer(n_inputs=256, n_neurons=128, length=4096, use_gpu=True)
```

## 5. Benchmarking

**Python v2-only suite** (pack, popcount, dense, GPU, scaling, memory):

```bash
python benchmarks/benchmark_suite.py --markdown   # writes benchmarks/results/BENCHMARKS.md
python benchmarks/benchmark_suite.py --full        # 10x iterations
```

**v2 vs v3 head-to-head** (all six forward variants, per-op speedup):

```bash
python examples/10_benchmark_report.py
```

**Rust-native Criterion** (no Python overhead):

```bash
cd engine
cargo bench --bench bitstream_bench
cargo bench --bench full_bench
cargo bench --bench scaling_bench
```

### Bitstream length scaling

Throughput (Mbit/s) is constant across L. The suite sweeps a 32x16 dense
layer at L = 128, 256, 512, 1024, 2048, 4096; wall time grows linearly.

### Batch size effects (`forward_batch_numpy`, 64->32, L=1024)

| Batch | Time | Regime |
|-------|------|--------|
| 1 | ~0.1 ms | FFI-dominated |
| 10 | ~0.3 ms | Rayon saturates |
| 100 | ~1.5 ms | linear |
| 1000 | ~15 ms | memory-bandwidth bound |

## 6. Rayon Thresholds

| Constant | Value | Effect |
|----------|-------|--------|
| `RAYON_ENCODE_THRESHOLD` | 128 | parallel input encoding when `n_inputs >= 128` |
| `RAYON_NEURON_THRESHOLD` | 8 | parallel accumulation when `n_neurons >= 8` |

Below these, sequential Rust loops avoid thread-pool synchronization cost.

## 7. Representative Speedups

AMD Zen 4 (AVX-512), Rust 1.82, Python 3.12:

| Operation | Python (ms) | Rust (ms) | Speedup |
|-----------|------------|-----------|---------|
| pack 1M bits | 12.3 | 0.18 | 68x |
| popcount 1M words | 8.7 | 0.09 | 97x |
| dense 64->32, L=1024 (list) | 45.2 | 0.52 | 87x |
| dense 64->32, L=1024 (numpy) | 45.2 | 0.31 | 146x |
| dense batch 100x64->32 | 4520 | 18.1 | 250x |
| LIF 100K steps (batch) | 312 | 0.74 | 422x |

On AVX2, expect 30-60% of AVX-512 throughput for popcount-dominated workloads.
ARM NEON provides 2-4x over portable scalar.
