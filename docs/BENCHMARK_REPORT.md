# SC-NeuroCore v3 Benchmark Report

**Version**: 3.1.0  
**Date**: 2026-02-10  
**SIMD Tier**: avx512-vpopcntdq

## Phase 7 Results (Dense Fast + Prepacked)

Measured via `examples/03_benchmark_report.py` on this machine.

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

## Phase 6 Results (Reference)

| Operation | v2 (ms) | v3 (ms) | Speedup | Target |
|-----------|---------|---------|---------|--------|
| pack (list, 1000K) | 8.035 | 35.092 | 0.2x | 6x |
| pack (numpy, 1000K) | 8.035 | 6.993 | 1.1x | 6x |
| popcount (list, 1000K) | 93.441 | 137.367 | 0.7x | 20x |
| popcount (numpy, 1000K) | 93.441 | 1.510 | 61.9x | 20x |
| dense forward (64->32, L=1024) | 2.795 | 2.064 | 1.4x | 70x |
| LIF (per-call, 100K) | 199.815 | 99.183 | 2.0x | 400x |
| LIF (batch, 100K) | 199.815 | 1.853 | 107.8x | 400x |

## Phase 5 Results (Reference)

| Operation | v2 (ms) | v3 (ms) | Speedup | Target |
|-----------|---------|---------|---------|--------|
| pack (1M bits) | 9.545 | 32.648 | 0.3x | 6x |
| popcount (1M words) | 97.481 | 141.040 | 0.7x | 20x |
| dense forward (64->32) | 3.018 | 1.041 | 2.9x | 70x |
| LIF step (100K) | 109.683 | 28.495 | 3.8x | 400x |

## Analysis

- `batch_encode_numpy + forward_prepacked` materially improves dense inference over baseline `forward` (`7.4x` vs `0.2x` on this run), validating the pre-packed path.
- `forward_fast` closes much of the baseline regression and remains deterministic per seed; the main gain comes when encoding can be amortized or skipped.
- `popcount_numpy` and `batch_lif_run` remain the strongest wins from the Phase 6/7 architecture (87.4x and 160.6x on this run).
- Absolute values are machine- and workload-dependent; the benchmark script is intended for relative trend tracking across phases.
