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
| Python models | **158 lazy-loaded classes / 152 source modules** | 11 | 6 | Custom eq. | 3 |
| Rust/compiled models | **175 Rust PyO3 wrappers / 161-model NetworkRunner** | — | — | C++ codegen | — |
| Hardware emulators | **9** | — | — | — | Loihi only |
| Formal verification | **18 SymbiYosys proof jobs and 130 formal statements (100 assert, 7 assume, 23 cover)** | — | — | — | — |
| Train-to-FPGA export | **Yes** | No | No | No | No |
