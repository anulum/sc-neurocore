# Yosys Synthesis Results

Gate-level synthesis of SC-NeuroCore Verilog RTL modules. The original rows use
Yosys 0.63; the current AER encoder observation uses Yosys 0.33 as detailed below.

**Synthesized**: 2026-03-25
**Yosys version**: 0.63+173 (OSS CAD Suite)
**Target**: Generic gate-level (no specific technology library)
**Command**: `yosys -p "read_verilog <module>.v; synth -flatten; stat"`

## Results

| Module | Cells | Wire Bits | Function |
|--------|------:|----------:|----------|
| `sc_bitstream_encoder.v` | 115 | 181 | LFSR-16 + comparator (bit-true with Python PredictiveSpikeCodec LFSR mode) |
| `sc_cordiv.v` | 2 | 6 | CORDIV stochastic division (1 AND + 1 MUX per bit) |
| `sc_dotproduct_to_current.v` | 448 | 515 | AND accumulation + popcount for dense layer |
| `sc_aer_encoder.v` | 1,953 | 2,132 | Corrected pending-bit priority encoder; Yosys 0.33, 2026-10-07 |
| `sc_event_neuron.v` | 2,135 | 2,202 | Event-driven LIF with async spike I/O |
| `sc_lif_neuron.v` | 3,134 | 3,216 | Q8.8 fixed-point LIF (integrate + fire + reset) |

## Current AER encoder observation — 2026-10-07

The corrected default encoder (`N_NEURONS=128`, seven-bit addresses and
16-bit timestamps) uses 1,953 generic cells and 2,132 wire bits with Yosys
0.33 (`2584903a060`). Repeating the same synthesis on the previous source
produced 1,959 cells and 2,138 wire bits: both counts decrease by six. The old
1,423-cell observation used Yosys 0.63 and a different source; it is historical
and cannot establish this comparison.

Actual RTL regressions cover all low-four-neuron masks, every address,
full-population bursts, simultaneous merge/coalescing, timestamp wrap and reset
at populations of four and 128 with eight- and 16-bit timestamps. The optimized
default netlist passes the same 148-case population trace. Its full burst emits
128 addresses over 128 service clocks after one capture clock. Yosys proves all
186 source/optimized equivalence cells after reset normalization.

The [source-bound report](https://github.com/anulum/sc-neurocore/blob/main/hdl/reports/yosys_aer_encoder_2026-10-07.json)
records the commands, counts and scope. These are generic synthesis and
simulation observations; board clock limits and rail energy need device
measurements.

## BCI Codec Hardware Estimates

For a 1024-channel predictive spike codec on-implant:

| Component | Per-Channel Cells | 1024 Channels |
|-----------|------------------:|--------------:|
| LFSR predictor (`sc_bitstream_encoder.v`) | 115 | 117,760 |
| XOR error gate | 1 | 1,024 |
| Q8.8 rate accumulator (shift-add) | ~80 | ~82,000 |
| ISI counter + varint encoder | ~200 | ~205,000 |
| **Total (excl. SRAM)** | **~396** | **~406K** |

At 7nm standard cell (~0.05 um^2/gate), 406K gates = ~0.02 mm^2.
Power estimate: ~2-5 mW at 1V, 20 kHz (dominated by SRAM for ISI FIFOs).

These are Yosys generic synthesis numbers. ASIC/FPGA-specific results
require technology-mapped synthesis (Synopsys DC, Vivado, Quartus).

## Notes

- `sc_cordiv.v` is trivially small (2 cells) because CORDIV is sequential — one AND gate and one MUX per clock cycle
- `sc_lif_neuron.v` is the largest because Q8.8 fixed-point arithmetic requires barrel shifters and adders for the membrane dynamics
- `sc_bitstream_encoder.v` matches Python `FixedPointLFSR` + `FixedPointBitstreamEncoder` and the Rust `Lfsr16` + `BitstreamEncoder` — bit-for-bit identical
