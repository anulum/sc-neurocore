<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Benchmark Comparison

Measured performance of SC-NeuroCore against competing frameworks.
All benchmarks are reproducible — see `benchmarks/` in the repository.

![Benchmark charts](../assets/benchmarks.png)

## Brunel Balanced Network (vs Brian2)

Standard 800E/200I Izhikevich network with random connectivity and
Poisson drive (Brunel 2000).

| Network size | SC-NeuroCore (Numba JIT) | Brian2 (C++ codegen) | Speedup |
|-------------|--------------------------|----------------------|---------|
| 1 000 neurons | 0.35 s | 1.38 s | **4.0x faster** |
| 10 000 neurons | 5.9 s | 4.4 s | 1.35x slower |

Firing rates match within 1% (100 Hz). SC-NeuroCore targets FPGA-scale
networks (≤5K neurons) where bit-exact RTL co-simulation matters.

## SNN Training (MNIST)

| Architecture | Method | Accuracy |
|-------------|--------|----------|
| FC 784→128→128→10 | Surrogate gradient (10 epochs) | 95.5% |
| FC 784→128→128→10 | + learnable τ (Fang 2021) | 97.7% |
| Conv→LIF→Pool→Conv→LIF→Pool→FC | Learnable β + threshold | **99.49%** (`benchmarks/results/mnist_conv_accuracy_reproducibility.json`) |

snnTorch achieves 95.8% on the same FC architecture (same setup,
10 epochs). The value of SC-NeuroCore is not training performance — it's
the `to_sc_weights()` export path to synthesisable FPGA hardware.

## Rust SIMD Engine

Criterion benchmarks on Intel i7-10700K:

| Operation | AVX-512 | AVX2 | Scalar |
|-----------|---------|------|--------|
| Bitstream packing | 113 Gbit/s | ~20 Gbit/s | ~3 Gbit/s |
| LIF neuron stepping | 456 Mstep/s | ~120 Mstep/s | ~40 Mstep/s |
| HDC query (100 patterns) | 1.1 ms | ~2 ms | ~8 ms |

Runtime SIMD detection selects the fastest available path. ARM NEON
supported for Apple Silicon and Raspberry Pi.

## FPGA Synthesis (Yosys)

| Configuration | LUTs (Xilinx 7-series) | Fmax (est.) |
|---------------|------------------------|-------------|
| sc_neurocore_top (default) | 3 673 | ~100 MHz |
| MNIST 16→10 (estimated) | ~56 000 | ~80 MHz |
| Target: Artix-7 100T | 63 400 available | — |

## Fault Tolerance

SC bitstreams degrade gracefully under random bit errors:

| Error rate | Accuracy loss (balanced p=0.5) |
|------------|-------------------------------|
| 1% | < 1% |
| 5% | < 1% |
| 10% | ~1% |
| 20% | ~2% |

This is a property of stochastic encoding (Alaghi & Hayes 2013), not
SC-NeuroCore-specific.

## Neuron Model Coverage

| Category | SC-NeuroCore | snnTorch | Norse | Brian2 | Lava |
|----------|:---:|:---:|:---:|:---:|:---:|
| Python models | **158 lazy-loaded classes / 153 source modules** | 11 | 6 | Custom eq. | 3 |
| Rust/compiled models | **175 Rust PyO3 wrappers / 162-model NetworkRunner** | — | — | C++ codegen | — |
| Hardware emulators | **9** | — | — | — | Loihi only |
| Formal verification | **50 SymbiYosys proof jobs and 180 formal statements (150 assert, 7 assume, 23 cover)** | — | — | — | — |
| Train-to-FPGA export | **Yes** | No | No | No | No |

## IQIF signed-integer polyglot batch loop

The committed `benchmarks/results/local_python_2026-07-14_iqif.json` records
the exact Wu et al. (2021) piecewise-linear integer recurrence through every
maintained backend. One 1,000-step warm-up precedes seven 200,000-step samples
per backend. The run was pinned to logical CPU 4, but the host had no isolated
CPU set; these are same-host regression timings, not portable speed claims.

| Backend | Median call | Median ns/step | Events | State mismatches | Final `v` |
|---|---:|---:|---:|---:|---:|
| Python | 92.358875 ms | 461.794375 | 13,333 | 0 | 165 |
| Rust | 2.437217 ms | 12.186085 | 13,333 | 0 | 165 |
| Julia | 5.550568 ms | 27.752840 | 13,333 | 0 | 165 |
| Go | 5.626361 ms | 28.131805 | 13,333 | 0 | 165 |
| Mojo | 2.261707 ms | 11.308535 | 13,333 | 0 | 165 |

All five trajectories have little-endian int64 SHA-256
`b5c84ffb7167e23d9ba3a1e4290aa93326649bd65087781e491a237ab347a4f4`.
The measured and dispatcher order is Mojo, Rust, Julia, Go, then Python. The
artifact binds the source and exact loaded Rust/Go/Mojo binaries, runtime
versions, affinity, governor, load averages, and raw timing samples. It is
neither a hardware measurement nor a production or cross-host performance
claim.

## McCulloch-Pitts logical batch loop

The committed
`benchmarks/results/local_python_2026-07-14_mcculloch_pitts.json` records the
source-faithful active-excitatory-count threshold and absolute-inhibition rule
through all five public dispatchers. Each call validates 200,000 varying rows
before execution, so the numbers include the shared Python-side public input
contract rather than timing an isolated inner comparator. One 1,000-row warm-up
precedes seven samples per backend.

| Backend | Median call | Median ns/row | Events | Trace mismatches |
|---|---:|---:|---:|---:|
| Rust | 234.741 ms | 1,173.704 | 102,273 | 0 |
| Go | 306.158 ms | 1,530.789 | 102,273 | 0 |
| Python | 328.712 ms | 1,643.559 | 102,273 | 0 |
| Mojo | 625.298 ms | 3,126.492 | 102,273 | 0 |
| Julia | 821.117 ms | 4,105.586 | 102,273 | 0 |

Every lane has binary-event SHA-256
`52a05b62f801b9a9856ccac9f6d79f2821d564239b85fd06d454d1d44e28aee4`.
The measured native order used by `auto` is Rust, Go, Mojo, then Julia, with
Python retained as the always-available floor. The run was pinned to logical
CPU 4 under the `powersave` governor, but load averages were high and the CPU
was not isolated. These timings are same-host regression evidence only, not a
hardware result, portable throughput claim, or cross-framework comparison.

## Sigmoid-rate exact-relaxation batch loop

The committed `benchmarks/results/local_python_2026-07-14_sigmoid_rate.json`
records the configurable scalar exact-relaxation equation through Python, the
Rust engine, Julia, Go, and Mojo. One 1,000-step warm-up precedes five
200,000-step samples per backend.

| Backend | Median call | Median ns/step | Trace mismatches | Maximum error |
|---|---:|---:|---:|---:|
| Python | 73.788 ms | 368.938 | 0 | 0 |
| Rust | 46.985 ms | 234.926 | 0 | 0 |
| Julia | 17.493 ms | 87.467 | 0 | 0 |
| Go | 97.270 ms | 486.350 | 0 | 0 |
| Mojo | 14.610 ms | 73.048 | 0 | `3.08e-14` |

Python, Rust, Julia, and Go share trace SHA-256
`5241be414683ce92ba9886c13c0a9f5ef84886d5d48ddda05fc892b72274e07d`.
Mojo remains within the declared `5e-12` absolute tolerance but has a distinct
binary trace hash. The host was pinned to logical CPU 4 without exclusive
isolation and reported high load averages. These figures are local diagnostic
regression evidence, not production, hardware, cross-host, or cross-framework
performance claims.

## Threshold-linear algebraic rate batch

The committed
`benchmarks/results/local_python_2026-07-14_threshold_linear_rate.json`
records the configurable `gain * max(0, current - theta)` transfer through all
five public dispatchers. Five 200,000-value samples follow a 1,000-value warm-
up. Every runtime produces the same little-endian float64 trace SHA-256,
`cdb90f105692311ba359cfbf0574faa23586215e1a253ddcad29276b9bf69402`.

| Backend | Median call | Median ns/evaluation | Trace mismatches |
|---|---:|---:|---:|
| Python | 1.621 ms | 8.107 | 0 |
| Mojo | 3.388 ms | 16.938 | 0 |
| Rust | 3.892 ms | 19.458 | 0 |
| Go | 9.824 ms | 49.122 | 0 |
| Julia | 12.425 ms | 62.123 | 0 |

The Python path uses a vectorised constant fill and was the shortest raw call.
The compiled dispatcher keeps Python as its always-available floor and orders
the measured native lanes Mojo, Rust, Go, then Julia. The run was pinned to one
logical CPU but not exclusively isolated, and its recorded load average was
high. These timings are local regression evidence, not production, cross-host,
hardware, or universal-ranking claims.

## Chialvo map polyglot batch loop

The committed `benchmarks/bench_chialvo_map.py` runs the same
500,000-iteration recurrence through every production lane. Its recorded
`benchmarks/results/bench_chialvo_map.json` contains the
source hashes, toolchain versions, CPU affinity, governor, load, parity, and
event counts. The 2026-07-11 run was pinned to logical CPU 4 on an Intel
i5-11600K under the `powersave` governor; the host reported no kernel-isolated
CPU set.

| Backend | Median call | Speed-up vs Python | Maximum trace difference | Events |
|---|---:|---:|---:|---:|
| Rust | 7.270 ms | 299.30x | `5.195e-12` | 12,935 |
| Julia | 9.576 ms | 227.22x | `1.736e-12` | 12,935 |
| Mojo | 11.373 ms | 191.32x | `6.839e-7` | 12,935 |
| Go | 20.524 ms | 106.01x | `3.542e-12` | 12,935 |
| Python | 2,175.866 ms | 1.00x | `0` | 12,935 |

These timings describe that recorded host and workload. They are used for the
host-matched fastest-first dispatcher, not presented as portable latency or
cross-framework claims.

## Medvedev first-return polyglot batch loop

The committed `benchmarks/bench_medvedev_map.py` measures the source-derived
slow-calcium first-return recurrence through all five production lanes. The
500,000-iteration, five-repeat record was pinned to logical CPU 4 on the same
Intel i5-11600K host under the `powersave` governor. The host reported no
kernel-isolated CPU set, and the artefact records workstation load rather than
claiming an isolated measurement.

| Backend | Median call | Speed-up vs Python | Maximum trace difference | Events |
|---|---:|---:|---:|---:|
| Julia | 8.230 ms | 29.52x | `0` | 375,000 |
| Rust | 10.799 ms | 22.50x | `0` | 375,000 |
| Mojo | 19.524 ms | 12.44x | `6.806e-14` | 375,000 |
| Go | 20.959 ms | 11.59x | `0` | 375,000 |
| Python | 242.929 ms | 1.00x | `0` | 375,000 |

The result, source hashes, runtime versions, affinity, parity, and final state
are committed in `benchmarks/results/bench_medvedev_map.json`. These are
host-specific batch timings, not portable single-step latency claims.

## Ibarz-Tanaka four-branch map parity horizon

The corrected `benchmarks/bench_ibarz_tanaka_map.py` measures the source-derived
Ibarz et al. (2007) recurrence over its committed 1,000-iteration `I=0.2`
parity horizon, using 21 calls per backend. The run was pinned to logical CPU 10
on the same i5-11600K host under the `powersave` governor. The host reported no
kernel-isolated CPU set and high concurrent load.

| Backend | Median call | Speed-up vs Python | Maximum trace difference | Events |
|---|---:|---:|---:|---:|
| Rust | 0.079827 ms | 11.28× | `0` | 33 |
| Mojo | 0.148605 ms | 6.06× | `6.883e-15` | 33 |
| Go | 0.242630 ms | 3.71× | `0` | 33 |
| Julia | 0.249030 ms | 3.62× | `0` | 33 |
| Python | 0.900612 ms | 1.00× | `0` | 33 |

Rust, Julia, and Go are bit-exact. Mojo is event-exact over the enrolled
horizon, but FMA-level differences can alter this sensitive map's branch
sequence over much longer runs; indefinite trajectory parity is not claimed.
The complete host metadata and source hashes are in
`benchmarks/results/bench_ibarz_tanaka_map.json`.
