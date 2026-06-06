<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Deploy an SNN on FPGA in 20 Minutes

This tutorial walks through the full SC-NeuroCore pipeline: train a digit
classifier in Python, simulate it with stochastic bitstreams (bit-exact match
to RTL), export hardware artefacts, and optionally run FPGA tool reports.

Use this tutorial as the first hardware-oriented onboarding path. It teaches
the project’s main evidence discipline: Python results, quantised arithmetic,
RTL, synthesis reports, and deployment claims are separate artefacts. Do not
turn a planning estimate into a power, timing, or production-throughput claim.

**Prerequisites**:

- Python 3.10+, `pip install sc-neurocore scikit-learn`
- Optional: [Yosys](https://github.com/YosysHQ/yosys) plus any required
  SystemVerilog frontend helpers for open-source resource reports
- Optional: Xilinx Vivado Design Suite (for Artix-7/Zynq implementation) or Lattice iCEcube2 (for iCE40)
- Optional: FPGA board (Artix-7 100T, Zynq-7020, or iCE40 for physical deployment)

## 1. Train and quantise (5 min)

```bash
python examples/mnist_fpga/demo.py
```

This loads sklearn's 8×8 digits dataset (1,797 images, 10 classes), applies
PCA to reduce 64 features → 16, trains a logistic regression classifier, and
quantises weights to Q8.8 fixed-point (the format used by the Verilog RTL).

The deterministic demo configuration currently reports:

```
  Float accuracy:     94.2%
  Q8.8 accuracy:      94.2%
  Max quantization error: 0.0019
```

The quantisation error is bounded by 1/256 ≈ 0.0039 — one LSB of Q8.8.
Classification accuracy is preserved because the weight magnitudes are small
relative to the representable range (−128 to +127.996).

## 2. Stochastic computing simulation (5 min)

The same script runs inference using LFSR-based bitstream encoding that
matches `hdl/sc_bitstream_encoder.v` cycle-exactly:

```
  SC accuracy:        94.0% (100 samples, L=1024)
```

SC accuracy matches Q8.8 at this task complexity. Stochastic multiplication
has variance proportional to 1/L; increasing `--stream-length 4096` can
improve accuracy on harder tasks at the cost of more clock cycles on FPGA.

The key insight: **what you simulate in Python is what the FPGA executes**.
Same LFSR polynomial (`x^16 + x^14 + x^13 + x^11 + 1`), same seed
assignment strategy (input: `0xACE1 + i*7`, weight: `0xBEEF + j*N*13 + i*13`),
same Q8.8 comparator logic.

## 3. Export Verilog weights

```bash
python examples/mnist_fpga/demo.py --export-verilog hdl/generated/mnist_weights.vh
```

This generates a `.vh` include file with `localparam` declarations for each
weight in the 10×16 matrix (160 values). The weights are synthesised into
LUT constants — no BRAM needed at this scale.

## 4. Synthesise with Yosys (3 min)

The `sc_dense_matrix_layer` module in `hdl/sc_dense_matrix_layer.v` implements
the full per-neuron dense layer: N_INPUTS shared encoders, N_NEURONS × N_INPUTS
weight encoders and AND synapses, N_NEURONS dot-product units and LIF neurons.

```bash
python tools/yosys_synth.py --module sc_dense_matrix_layer --markdown
```

Treat this output as evidence only when the `Status` column is `OK`. If the
local Yosys build reports `SKIP` because a SystemVerilog frontend helper is
missing or a library module cannot be parsed, stop at the scaffold/export stage
and do not report LUT/FF numbers from that run.

The MNIST demo also prints a sizing estimate scaled from an older reference
configuration. That estimate is useful for planning target class, but it is not
a replacement for a tool-generated utilisation report.

## 5. Vivado implementation (optional, 5 min)

If you have Vivado installed, run the full implementation flow for timing and
power numbers:

```bash
vivado -mode batch -source tools/vivado_impl.tcl \
       -tclargs -top sc_dense_matrix_layer -part xc7a100tcsg324-1 -clk 250
```

Parse the reports:

```bash
python tools/vivado_report.py vivado_reports/
```

This gives measured Fmax, LUT/FF/BRAM utilisation, and dynamic power for the
specified part and clock. Use those generated reports, not estimates, when
publishing power or resource claims.

**Latency per inference**: `stream_len` clock cycles. At 250 MHz with L=1024:
1024 / 250 MHz = **4.1 µs per classification**.

## 6. Planning guide

The table below is a planning guide, not a synthesis report. Re-run the demo
and a target-specific synthesis flow before publishing resource or power
numbers.

| Configuration | Planning LUT estimate | Target FPGA class | Accuracy source |
|--------------|----------:|-------------|----------|
| 16→10 (PCA) | ~56K | Artix-7 100T | demo run |
| 8→10 (PCA) | ~28K | Artix-7 35T | rerun required |
| 64→10 (raw pixels) | ~225K | Kintex-7 325T | rerun required |
| 16→64→10 (2-layer) | ~400K | Virtex-7 | rerun required |

Resource usage scales as O(N_INPUTS × N_NEURONS) for a single layer.
Time-multiplexed designs (1 encoder cycling through N inputs) reduce LUTs
by N× at the cost of N× more clock cycles.

## What you proved

1. **Bit-exact simulation**: Python SC model matches Verilog RTL (same LFSR
   seeds, Q8.8 arithmetic, overflow semantics)
2. **Quantisation preserves accuracy**: Q8.8 introduces <0.004 max error,
   no accuracy loss for this task
3. **SC overhead is bounded**: accuracy gap = f(1/L), controllable via
   bitstream length
4. **Hardware handoff**: generated artefacts and latency formula are ready for
   a real synthesis or implementation flow; resource and power claims require
   tool-generated reports
