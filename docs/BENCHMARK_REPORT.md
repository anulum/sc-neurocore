# SC-NeuroCore v3 Benchmark Report

**Version**: 3.2.0  
**Date**: 2026-02-10  
**SIMD Tier**: avx512-vpopcntdq

## Phase 8 Results (forward_numpy + Parallel batch_encode_numpy)

Measured via `examples/03_benchmark_report.py` on this machine.

| Operation | v2 (ms) | v3 (ms) | Speedup | Target |
|-----------|---------|---------|---------|--------|
| pack (list, 1000K) | 12.983 | 35.539 | 0.4x | 6x |
| pack (numpy, 1000K) | 12.983 | 4.866 | 2.7x | 6x |
| popcount (list, 1000K) | 128.906 | 190.677 | 0.7x | 20x |
| popcount (numpy, 1000K) | 128.906 | 1.409 | 91.5x | 20x |
| dense forward (64->32, L=1024) | 3.982 | 5.804 | 0.7x | 70x |
| dense fast (64->32, L=1024) | 3.982 | 9.056 | 0.4x | 70x |
| dense prepacked (64->32, L=1024) | 3.982 | 4.367 | 0.9x | 70x |
| dense numpy (64->32, L=1024) | 3.982 | 10.873 | 0.4x | 70x |
| LIF (per-call, 100K) | 196.586 | 65.651 | 3.0x | 400x |
| LIF (batch, 100K) | 196.586 | 1.606 | 122.4x | 400x |

## Criterion Diagnosis (Phase 8)

Measured via `cargo bench` in `engine/`:

| Benchmark | Time (95% CI) |
|-----------|---------------|
| bernoulli_stream_1024 | 4.8035 µs - 5.6242 µs |
| bernoulli_stream_pack_1024 | 5.7678 µs - 6.5472 µs |
| bernoulli_packed_1024 | 5.4900 µs - 6.0629 µs |
| dense_forward_64x32 | 4.9936 ms - 6.8809 ms |
| dense_forward_fast_64x32 | 2.5554 ms - 3.6797 ms |
| dense_forward_prepacked_64x32 | 398.59 µs - 645.89 µs |

Interpretation:
- `bernoulli_packed_1024` is slightly faster than `bernoulli_stream_pack_1024` on this run, so direct packed generation itself does not explain dense forward regression.
- Dense timings confirm the expected ordering: `forward_prepacked` fastest, then `forward_fast`, then baseline `forward`.
- `forward` vs `forward_fast` gap is consistent with sequential vs parallel input encoding cost.

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
