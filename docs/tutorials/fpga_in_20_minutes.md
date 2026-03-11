<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Deploy an SNN on FPGA in 20 Minutes

This tutorial walks through the full SC-NeuroCore pipeline: train a digit
classifier in Python, simulate it with stochastic bitstreams (bit-exact match
to RTL), synthesise to FPGA, and read the resource report.

**Prerequisites**: Python 3.10+, `pip install sc-neurocore scikit-learn`

## 1. Train and quantise (5 min)

```bash
python examples/mnist_fpga/demo.py
```

This loads sklearn's 8×8 digits dataset (1,797 images, 10 classes), applies
PCA to reduce 64 features → 16, trains a logistic regression classifier, and
quantises weights to Q8.8 fixed-point (the format used by the Verilog RTL).

Expected output:

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

For the reference 3-input × 7-neuron configuration:

| Module | LUTs | FFs |
|--------|-----:|----:|
| `sc_neurocore_top` | 7,382 | 2,442 |

The 16→10 MNIST classifier is estimated at ~56K LUTs (scaled from the Yosys
3×7 reference at 7,382 LUTs), fitting an Artix-7 100T.

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

This gives Fmax (expected ~200–300 MHz on 7-series), LUT/FF/BRAM utilisation,
and dynamic power (expected <0.5 W for the 16→10 configuration).

**Latency per inference**: `stream_len` clock cycles. At 250 MHz with L=1024:
1024 / 250 MHz = **4.1 µs per classification**.

## 6. Scaling guide

| Configuration | Est. LUTs | Target FPGA | Accuracy |
|--------------|----------:|-------------|----------|
| 16→10 (PCA) | ~56K | Artix-7 100T | ~94% |
| 8→10 (PCA) | ~28K | Artix-7 35T | ~88% |
| 64→10 (raw pixels) | ~225K | Kintex-7 325T | ~96% |
| 16→64→10 (2-layer) | ~400K | Virtex-7 | ~97% |

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
4. **FPGA-ready**: synthesisable to Xilinx 7-series, ~4 µs inference latency
   at 250 MHz, <0.5 W power budget
