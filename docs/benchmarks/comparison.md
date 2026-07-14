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
| Rust/compiled models | **175 Rust PyO3 wrappers / 161-model NetworkRunner** | — | — | C++ codegen | — |
| Hardware emulators | **9** | — | — | — | Loihi only |
| Formal verification | **48 SymbiYosys proof jobs and 176 formal statements (146 assert, 7 assume, 23 cover)** | — | — | — | — |
| Train-to-FPGA export | **Yes** | No | No | No | No |

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
