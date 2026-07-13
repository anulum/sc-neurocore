<!-- © 1998–2026 Miroslav Šotek. All rights reserved. -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- ORCID: https://orcid.org/0009-0009-3560-0851 -->
<!-- License: GNU AFFERO GENERAL PUBLIC LICENSE v3 -->
<!-- Commercial Licensing: Available -->

# SC-NeuroCore v3 Benchmark Report

**Version**: 3.15.8
**Date**: 2026-04-13 (core engine benchmarks from 2026-03-15, FPGA added)
**Previous**: 3.6.0 (2026-02-10)
**SIMD Tier**: avx512-vpopcntdq

## Baseline Definition and Routing Note

- `v2` in this report means the SC-NeuroCore v2 Python reference path measured by the same benchmark harness.
- External framework baselines (Norse/Sinabs/Lava CPU) are not yet included in this file and must be added before making ecosystem-level claims.
- For low-latency use (single sample or micro-batch), prefer `DenseLayer.forward_fast`.
- For throughput use (batch >= 10), prefer `DenseLayer.forward_batch_numpy`.
- This report is release evidence only for rows backed by committed benchmark
  artefacts or named tool reports. New local benchmark numbers must not be
  promoted into public claims until the raw JSON, CSV, or companion paper
  artefact plus environment provenance is committed.

## Fused Dense, Fast PRNG, and Batch Forward Results

Measured via `examples/03_benchmark_report.py` on this machine.

| Operation | v2 (ms) | v3 (ms) | Speedup | Target |
|-----------|---------|---------|---------|--------|
| pack (list, 1000K) | 11.538 | 35.448 | 0.3x | 6x |
| pack (numpy, 1000K) | 11.538 | 0.129 | 89.3x | 6x |
| popcount (list, 1000K) | 109.023 | 151.322 | 0.7x | 20x |
| popcount (numpy, 1000K) | 109.023 | 1.989 | 54.8x | 20x |
| dense forward (64->32, L=1024) | 3.728 | 1.598 | 2.3x | 70x |
| dense fast (64->32, L=1024) | 3.728 | 0.299 | 12.4x | 70x |
| dense prepacked (64->32, L=1024) | 3.728 | 0.282 | 13.2x | 70x |
| dense prepacked numpy (64->32, L=1024) | 3.728 | 0.110 | 33.9x | 70x |
| dense numpy (64->32, L=1024) | 3.728 | 0.647 | 5.8x | 70x |
| dense fused (64->32, L=1024) | 4.664 | 0.380 | 12.3x | 70x |
| dense batch (100x64->32, L=1024) | 289.305 | 6.893 | 42.0x | 70x |
| LIF (per-call, 100K) | 126.313 | 25.525 | 4.9x | 400x |
| LIF (batch, 100K) | 126.313 | 0.905 | 139.6x | 400x |
| LIF multi (100x100K) | 12911.296 | 25.196 | 512.4x | 400x |

## Criterion Diagnosis for Fused Dense and Fast PRNG

Measured via targeted commands:

```powershell
cargo bench --bench full_bench dense_forward_fused
cargo bench --bench full_bench encode_and_popcount
cargo bench --bench full_bench dense_forward_batch
cargo bench --bench full_bench prng_xoshiro
```

| Benchmark | Time (95% CI) |
|-----------|---------------|
| dense_forward_fused_64x32 | 1.1268 ms - 1.9825 ms |
| bernoulli_encode_and_popcount_1024 | 342.59 ns - 408.10 ns |
| dense_forward_batch_64x32_x100 | 21.842 ms - 28.753 ms |
| prng_xoshiro_fill_1024 | 1.5879 us - 1.7596 us |

Interpretation:
- Fused encode+AND+popcount path is functionally correct and benchmarked end-to-end.
- Batched dense API reduces Python-level overhead substantially vs per-sample loops.
- Multi-neuron LIF remains above the Blueprint 400x target on this host.

## SIMD Dense Inner Loop Results (Reference)

| Operation | v2 (ms) | v3 (ms) | Speedup | Target |
|-----------|---------|---------|---------|--------|
| pack (list, 1000K) | 10.337 | 37.799 | 0.3x | 6x |
| pack (numpy, 1000K) | 10.337 | 0.069 | 149.3x | 6x |
| popcount (list, 1000K) | 96.956 | 135.444 | 0.7x | 20x |
| popcount (numpy, 1000K) | 96.956 | 1.563 | 62.0x | 20x |
| dense forward (64->32, L=1024) | 2.953 | 0.683 | 4.3x | 70x |
| dense fast (64->32, L=1024) | 2.953 | 0.171 | 17.3x | 70x |
| dense prepacked (64->32, L=1024) | 2.953 | 0.092 | 31.9x | 70x |
| dense prepacked numpy (64->32, L=1024) | 2.953 | 0.033 | 90.2x | 70x |
| dense numpy (64->32, L=1024) | 2.953 | 0.118 | 25.1x | 70x |
| LIF (per-call, 100K) | 106.451 | 23.925 | 4.4x | 400x |
| LIF (batch, 100K) | 106.451 | 0.897 | 118.7x | 400x |
| LIF multi (100x100K) | 13349.151 | 31.783 | 420.0x | 400x |

## SIMD Pack Dispatch Results (Reference)

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

## Fast Bernoulli Encoding Results (Reference)

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

## Dense Forward Optimization Results (Reference)

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

## Bioware modularisation diagnostics

`benchmarks/results/bench_bioware.json` records 30 interleaved cold-process
samples for parent `c4e492ff5` and the modular Bioware candidate. The probe
executes a deterministic 50 ms, eight-channel MEA frame through spike
detection, 16-bit AER, deterministic LFSR bitstreams, optical pulse proposals,
culture analysis, LFP bands, and the legacy fitness adapter.

Both variants produced exactly 6,865 canonical bytes with SHA-256
`2491dc73a2de93a45a1cc944539c170b151403e42b973b18806143f318b7d669`
(7 spikes, 7 AER events, 4 bitstreams, and 4 pulses).

| Local diagnostic metric | Parent median | Modular candidate median | Delta |
| --- | ---: | ---: | ---: |
| Pipeline | 2.801 ms | 3.345 ms | +19.41% |
| Import | 34.771 ms | 50.265 ms | +44.56% |
| Subprocess wall | 579.937 ms | 535.945 ms | -7.59% |
| Maximum RSS | 37,076 KiB | 37,154 KiB | +0.21% |

The run used taskset affinity but not a kernel-reserved isolated core. The
workstation was under concurrent load and its governor was `powersave`.
Timings are local regression context only, not throughput or hardware claims.
The maintained full pipeline is Python-only; removed generated Go, Julia,
Mojo, and Rust placeholders were not executable backends. Independent Julia
plasticity solvers remain outside this orchestration benchmark.

## Autonomous-learning modularisation diagnostics

`benchmarks/results/bench_learning_bridge.json` records five interleaved
cold-process samples for parent `5f3d5d254` and the modular learning-bridge
candidate. Each sample executes 1,024 deterministic STDP events through the
Python-to-Rust scalar and batched wrappers, a Rayon layer, the Torch transition,
the Go cgo adapter, and the Julia C-FFI adapter. The evidence hashes both source
trees and both exact native libraries; it records no local filesystem paths.

Every path converged to the same weight (Rust/Torch
`0.7806524634361267`, Go `0.780652463`, Julia `0.78065246`). Parent and
candidate produced the same 314-byte normalized canonical payload with SHA-256
`020da9895ff4d7f1150fd7159c30e7c476afe9ff46c6d652bf0ad73fa6cf309d`.
Rayon weight and opaque-state hashes were also identical.

| Local diagnostic metric | Parent median | Modular candidate median | Delta |
| --- | ---: | ---: | ---: |
| Cold import | 4,793.066 ms | 4,627.645 ms | -3.45% |
| Rust scalar, 1,024 calls | 3.538 ms | 10.845 ms | +206.51% |
| Rust batched, 1,024 events | 0.586 ms | 0.924 ms | +57.53% |
| Rayon layer, 1,024 synapses | 1.522 ms | 2.296 ms | +50.85% |
| Torch, 1,024 updates | 380.845 ms | 448.397 ms | +17.74% |
| Go process | 1,190.849 ms | 1,316.371 ms | +10.54% |
| Julia process | 942.785 ms | 995.434 ms | +5.58% |
| Subprocess wall | 10,261.711 ms | 9,640.037 ms | -6.06% |
| Maximum RSS | 658,592 KiB | 660,072 KiB | +0.22% |

These timings were captured with taskset affinity but without an isolated core.
Host load was 29.17/29.63/36.99 before and 27.88/28.42/35.54 after the run, so small-kernel
medians are scheduler-sensitive and must not be used as throughput or release
promotion claims. The scalar candidate additionally performs fail-closed
Boolean, finite-value, rule-domain, and timestep validation that the parent
omitted; production event arrays should use the checked batched or layer APIs
to amortize FFI crossings. A publishable performance claim requires a rerun on
reserved isolated cores with recorded governor, frequency, versions, and idle
load. The committed result is correctness and local regression evidence only.
