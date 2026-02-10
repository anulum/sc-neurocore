# SC-NeuroCore v3 Benchmark Report

**Version**: 3.3.0  
**Date**: 2026-02-10  
**SIMD Tier**: avx512-vpopcntdq

## Phase 9 Results (Fast Bernoulli + Fused AND+Popcount + Zero-Copy Prepacked)

Measured via `examples/03_benchmark_report.py` on this machine.

| Operation | v2 (ms) | v3 (ms) | Speedup | Target |
|-----------|---------|---------|---------|--------|
| pack (list, 1000K) | 10.807 | 62.841 | 0.2x | 6x |
| pack (numpy, 1000K) | 10.807 | 9.415 | 1.1x | 6x |
| popcount (list, 1000K) | 118.885 | 144.767 | 0.8x | 20x |
| popcount (numpy, 1000K) | 118.885 | 1.866 | 63.7x | 20x |
| dense forward (64->32, L=1024) | 6.971 | 8.034 | 0.9x | 70x |
| dense fast (64->32, L=1024) | 6.971 | 6.125 | 1.1x | 70x |
| dense prepacked (64->32, L=1024) | 6.971 | 3.599 | 1.9x | 70x |
| dense prepacked numpy (64->32, L=1024) | 6.971 | 0.085 | 81.6x | 70x |
| dense numpy (64->32, L=1024) | 6.971 | 4.908 | 1.4x | 70x |
| LIF (per-call, 100K) | 143.202 | 35.008 | 4.1x | 400x |
| LIF (batch, 100K) | 143.202 | 1.404 | 102.0x | 400x |

## Criterion Diagnosis (Phase 9)

Measured via `cargo bench` in `engine/`:

| Benchmark | Time (95% CI) |
|-----------|---------------|
| bernoulli_stream_1024 | 5.1501 µs - 5.9660 µs |
| bernoulli_stream_pack_1024 | 6.5709 µs - 7.3970 µs |
| bernoulli_packed_1024 | 4.9971 µs - 5.9171 µs |
| bernoulli_packed_fast_1024 | 2.2928 µs - 2.6434 µs |
| dense_forward_64x32 | 4.4892 ms - 5.7135 ms |
| dense_forward_fast_64x32 | 3.6861 ms - 5.4961 ms |
| dense_forward_prepacked_64x32 | 364.30 µs - 539.50 µs |

Interpretation:
- `bernoulli_packed_fast_1024` is ~2.2x faster than `bernoulli_packed_1024` in this environment.
- `dense_forward_prepacked_64x32` remains the fastest dense path in criterion runs.
- New `dense prepacked numpy` in Python benchmarks reaches 81.6x vs v2, confirming the zero-copy prepacked path removes most Python-side overhead.

## Phase 7 Results (Reference)

| Operation | v2 (ms) | v3 (ms) | Speedup | Target |
|-----------|---------|---------|---------|--------|
| pack (list, 1000K) | 15.208 | 54.526 | 0.3x | 6x |
| pack (numpy, 1000K) | 15.208 | 10.315 | 1.5x | 6x |
| popcount (list, 1000K) | 108.495 | 316.783 | 0.3x | 20x |
| popcount (numpy, 1000K) | 108.495 | 1.242 | 87.4x | 20x |
| dense forward (64->32, L=1024) | 4.173 | 20.570 | 0.2x | 70x |
| dense fast (64->32, L=1024) | 4.173 | 4.318 | 1.0x | 70x |
| dense prepacked (64->32, L=1024) | 4.173 | 0.562 | 7.4x | 70x |
| LIF (per-call, 100K) | 240.266 | 61.585 | 3.9x | 400x |
| LIF (batch, 100K) | 240.266 | 1.496 | 160.6x | 400x |
