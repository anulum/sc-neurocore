# SC-NeuroCore v3 Benchmark Report

**Version**: 3.0.0
**Date**: 2026-02-10
**SIMD Tier**: avx512-vpopcntdq

## Phase 6 Results (NumPy Zero-Copy + Batch)

| Operation | v2 (ms) | v3 (ms) | Speedup | Target |
|-----------|---------|---------|---------|--------|
| pack (list, 1000K) | 8.035 | 35.092 | 0.2x | 6x |
| pack (numpy, 1000K) | 8.035 | 6.993 | 1.1x | 6x |
| popcount (list, 1000K) | 93.441 | 137.367 | 0.7x | 20x |
| popcount (numpy, 1000K) | 93.441 | 1.510 | 61.9x | 20x |
| dense forward (64->32, L=1024) | 2.795 | 2.064 | 1.4x | 70x |
| LIF (per-call, 100K) | 199.815 | 99.183 | 2.0x | 400x |
| LIF (batch, 100K) | 199.815 | 1.853 | 107.8x | 400x |

## Phase 5 Results (Reference — List-Based FFI)

| Operation | v2 (ms) | v3 (ms) | Speedup | Target |
|-----------|---------|---------|---------|--------|
| pack (1M bits) | 9.545 | 32.648 | 0.3x | 6x |
| popcount (1M words) | 97.481 | 141.040 | 0.7x | 20x |
| dense forward (64->32) | 3.018 | 1.041 | 2.9x | 70x |
| LIF step (100K) | 109.683 | 28.495 | 3.8x | 400x |

## Analysis

NumPy zero-copy and batch APIs removed the dominant Python list -> Rust Vec marshalling overhead for the targeted paths.

- `popcount_numpy` now exceeds Blueprint section 8 target (`61.9x` vs `20x`) on this machine.
- `batch_lif_run` improves from per-call `2.0x` to `107.8x`, showing the FFI boundary cost was the main bottleneck, but it still falls short of the `400x` target.
- `pack_bitstream_numpy` improves over list mode (`1.1x` vs `0.2x`) but remains below the `6x` target.
- `dense forward` remains below target (`1.4x` vs `70x`) and likely needs deeper kernel-level optimization and/or target definition alignment.

Overall, the Phase 6 changes validate the intended direction: zero-copy and batch execution unlock substantial speedups where boundary overhead dominated, while some Blueprint targets remain aspirational for this workload and environment.
