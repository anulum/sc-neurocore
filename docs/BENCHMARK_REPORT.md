# SC-NeuroCore v3 Benchmark Report

**Version**: 3.4.0  
**Date**: 2026-02-10  
**SIMD Tier**: avx512-vpopcntdq

## Phase 10 Results (SIMD Pack + LIF Multi + Rayon Guard)

Measured via `examples/03_benchmark_report.py` on this machine.

| Operation | v2 (ms) | v3 (ms) | Speedup | Target |
|-----------|---------|---------|---------|--------|
| pack (list, 1000K) | 16.918 | 45.964 | 0.4x | 6x |
| pack (numpy, 1000K) | 16.918 | 0.133 | 127.0x | 6x |
| popcount (list, 1000K) | 94.333 | 138.951 | 0.7x | 20x |
| popcount (numpy, 1000K) | 94.333 | 1.303 | 72.4x | 20x |
| dense forward (64->32, L=1024) | 7.077 | 19.442 | 0.4x | 70x |
| dense fast (64->32, L=1024) | 7.077 | 17.781 | 0.4x | 70x |
| dense prepacked (64->32, L=1024) | 7.077 | 5.453 | 1.3x | 70x |
| dense prepacked numpy (64->32, L=1024) | 7.077 | 6.125 | 1.2x | 70x |
| dense numpy (64->32, L=1024) | 7.077 | 6.727 | 1.1x | 70x |
| LIF (per-call, 100K) | 139.417 | 27.015 | 5.2x | 400x |
| LIF (batch, 100K) | 139.417 | 0.992 | 140.5x | 400x |
| LIF multi (100x100K) | 15442.319 | 90.480 | 170.7x | 400x |

## Criterion Diagnosis (Phase 10)

Measured via targeted commands:

```powershell
cargo bench --bench full_bench pack_1m
cargo bench --bench full_bench pack_fast_1m
cargo bench --bench full_bench pack_dispatch_1m
cargo bench --bench full_bench lif_10k_steps
cargo bench --bench full_bench lif_100k_steps
```

| Benchmark | Time (95% CI) |
|-----------|---------------|
| pack_1m | 1.0666 ms - 1.2110 ms |
| pack_fast_1m | 485.91 us - 554.76 us |
| pack_dispatch_1m | 33.289 us - 41.916 us |
| lif_10k_steps | 31.737 us - 34.811 us |
| lif_100k_steps | 341.93 us - 390.05 us |

Interpretation:
- SIMD dispatch closes the pack target by a wide margin on numpy path.
- `pack_dispatch_1m` is ~27x faster than baseline `pack_1m`.
- Single-neuron batch LIF improved vs Phase 9, and multi-neuron parallel batch further increases throughput.

## Phase 9 Results (Reference)

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
