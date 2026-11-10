<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Performance tuning: precision, throughput and what is measured
-->

# Performance Tuning

This page lists the settings that change SC-NeuroCore's precision and speed,
and the evidence behind each one. Measure before and after every change.

- For pytest timing guards, use the maintained
  [Performance-Gated Tests](performance_gates.md) selector. Those tests are
  small wall-clock regression checks.
- A published speed figure needs its raw benchmark artefact and must follow
  the [benchmark evidence rules](../benchmarks/cross_framework.md#evidence-rules).

## 1. Bitstream length and precision

A value with probability `p` is encoded as a bitstream of `N` independent
Bernoulli bits and read back as the fraction of ones. That estimate is
unbiased. Its standard deviation is `sqrt(p (1 - p) / N)`, which is largest at
`p = 0.5`, where it is `1 / (2 sqrt(N))`:

| Bitstream length `N` | Largest standard deviation |
|----------------------|----------------------------|
| 256 | 0.031 |
| 1,024 | 0.016 |
| 4,096 | 0.0078 |
| 16,384 | 0.0039 |

These numbers follow from the binomial distribution. They are not
measurements of a layer.

### Choosing a length

`sc_neurocore.utils.bitstreams.adaptive_length(p, epsilon, confidence, method)`
returns the shortest length that keeps the error below `epsilon`. The length is
rounded up to a power of two. For `p = 0.5` and a confidence of 0.95 it returns:

| Error bound `epsilon` | `"hoeffding"` | `"chebyshev"` | `"variance"` |
|-----------------------|---------------|---------------|--------------|
| 0.05 | 1,024 | 2,048 | 128 |
| 0.02 | 8,192 | 16,384 | 1,024 |
| 0.01 | 32,768 | 65,536 | 4,096 |

`"variance"` makes the standard deviation equal `epsilon` and ignores the
confidence level. The other two hold with the stated confidence.

For a whole network, `sc_neurocore.compiler.adaptive_precision.assign_lengths`
assigns one length per layer from Hoeffding or sensitivity bounds. See
[Adaptive Precision](../api/adaptive_precision.md).

## 2. Software throughput

- **Packed bitwise layers.** `sc_neurocore.layers.vectorized_layer.VectorizedSCLayer`
  packs bitstreams into 64-bit words with
  `sc_neurocore.accel.vector_ops.pack_bitstream`. One word operation therefore
  processes 64 bits. The layer has three settings:
  - with CuPy installed and `use_gpu=True` (the default), it runs on a GPU;
    otherwise it uses NumPy;
  - `sparse=True` stores the connectivity as a SciPy sparse matrix; `connectivity`
    sets the fraction of connections kept;
  - `length` sets the bitstream length; see section 1.

  See [Layers](../api/layers.md#vectorized-layer).
- **Rust engine.** The compiled `sc_neurocore_engine` runs SC inference with
  SIMD. See [Rust Engine & Performance](../tutorials/05_rust_engine_performance.md).
- **GPU dense layer.** `sc_neurocore_engine.GpuDenseLayer` is built with the
  Rust feature `gpu`. It runs the SC dense layer through wgpu on Vulkan, Metal
  or DX12. See [GPU Compute Backend](../api/gpu.md).

The fastest path depends on the layer sizes and the hardware. The committed
comparisons are in the [benchmark report](../benchmarks/BENCHMARK_REPORT.md).
Measure your own case before choosing.

## 3. FPGA clock

There is one committed hardware timing result: the out-of-context Vivado
synthesis of the SHD core `sc_shd_top` for the Zynq XC7Z020 at 100 MHz. It was
run without weights or delays. `ARCHITECTURE.md`, section "SHD FPGA Pipeline",
gives the figures and the report files.

No other clock, device or design has a committed timing report, and no
design has been run on a board. Any clock target, even 100 MHz on a board, is
untested for SC-NeuroCore hardware.

## 4. Power

No power measurement of any SC-NeuroCore design exists.

`hdl/tb_power_comparison.v` and `examples/power_comparison.py` compare register
toggle counts of a clock-driven and an event-driven neuron. Toggles are one
input of dynamic power. Clock distribution, counters and control logic are not
counted, so the comparisons are not power figures. The
[event-driven tutorial](../tutorials/65_event_driven.md) gives the background.

## 5. Order of work

1. Measure the current state with the performance-gated tests or a benchmark
   harness, and keep the raw output.
2. Choose the bitstream length from the error you need (section 1).
3. Change one setting at a time and measure again.
4. Publish a figure only together with the artefact it comes from.
