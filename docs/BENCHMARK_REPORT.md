# SC-NeuroCore v3 Benchmark Report

- Date: 2026-02-10
- Platform: win32
- SIMD tier: avx512-vpopcntdq
- Version: 3.0.0-rc.1
- Script: `examples/03_benchmark_report.py`

## Results

| Operation | v2 (ms) | v3 (ms) | Speedup | Target |
|---|---:|---:|---:|---:|
| pack_bitstream (1000K bits) | 9.545 | 32.648 | 0.3x | 6x |
| popcount (1000K words) | 97.481 | 141.040 | 0.7x | 20x |
| dense forward (64->32, L=1024) | 3.018 | 1.041 | 2.9x | 70x |
| LIF step (100K steps) | 109.683 | 28.495 | 3.8x | 400x |

## Comparison To Blueprint Section 8 Targets

- `pack_bitstream`: target not met (`0.3x` vs `6x`).
- `popcount`: target not met (`0.7x` vs `20x`).
- `dense forward`: target not met (`2.9x` vs `70x`).
- `LIF step`: target not met (`3.8x` vs `400x`).

## Analysis

- Current benchmark path includes Python-level marshalling overhead (`list` conversion in the script) that dominates `pack_bitstream` and `popcount` for these shapes.
- Dense and LIF operations show meaningful kernel speedup over v2, but remain below blueprint targets in this single-threaded Python invocation path.
- Release-candidate validation goal for Phase 5 is satisfied (formal measured report produced); target tuning and deeper profiling remain follow-up work.

## Multi-Core Scaling Notes

- The script is single-threaded at the callsite.
- Rust kernels can benefit from rayon-enabled parallel execution; expected gains are workload and core-count dependent.
- Follow-up benchmarking should include pinned-core and multi-thread runs to quantify end-to-end scaling separately from Python call overhead.
