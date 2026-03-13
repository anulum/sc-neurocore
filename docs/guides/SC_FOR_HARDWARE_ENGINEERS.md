<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Stochastic Computing for Hardware Engineers

A guide for FPGA/ASIC designers who want to understand SC-NeuroCore's
hardware architecture, synthesis flow, and design trade-offs.

## SC arithmetic on silicon

Stochastic computing replaces complex arithmetic circuits with
trivial logic gates:

| Operation | Conventional (16-bit) | SC | Gate reduction |
|-----------|----------------------|------|----------------|
| Multiply | 256 LUTs (array mult) | 1 LUT (AND) | 256× |
| Add (scaled) | 16 LUTs (ripple adder) | 2 LUTs (MUX) | 8× |
| Integrate | ~50 LUTs (accumulator) | ~4 LUTs (counter) | 12× |
| Square | 256 LUTs | 0 (wire, auto-correlation) | ∞ |

The trade-off: SC requires L clock cycles per operation (L = bitstream
length). For L=256, an SC multiplier is 256 cycles × 1 LUT vs 1 cycle
× 256 LUTs. Same throughput, but SC uses 256× less area and power.

## SC-NeuroCore HDL modules

All Verilog modules are in `hdl/` and have been synthesised with
Yosys for Lattice iCE40 and ECP5:

| Module | Function | LUTs (iCE40) | FFs |
|--------|----------|-------------|-----|
| `sc_lif_neuron.v` | Q8.8 LIF neuron | ~120 | ~48 |
| `sc_stochastic_encoder.v` | Probability → bitstream | ~40 | ~16 |
| `sc_stochastic_decoder.v` | Bitstream → count | ~20 | ~16 |
| `sc_lfsr_16bit.v` | 16-bit LFSR (PRNG) | ~8 | 16 |
| `sc_synapse.v` | AND-gate multiply | 1 | 0 |
| `sc_mux_adder.v` | 2-input MUX (scaled add) | 2 | 0 |
| `sc_dense_layer_top.v` | N-input, M-neuron layer | ~N×M×2 | ~M×48 |
| `sc_bitstream_mux.v` | N-input MUX tree | ~N | ~log₂(N) |
| `sc_spike_counter.v` | Population spike count | ~M×8 | ~M×8 |
| `sc_noise_injector.v` | LFSR-based noise source | ~12 | 16 |

## Architecture: `sc_dense_layer_top.v`

The core building block is a fully-connected layer:

```
                ┌─────────────────────────────────┐
  N inputs ──►  │  N×M AND gates (weight multiply) │
                │  M MUX trees (weighted sum)       │
                │  M LIF neurons (integrate-fire)   │
                │  M spike outputs                  │
                └─────────────────────────────────┘
```

### Port map

```verilog
module sc_dense_layer_top #(
    parameter N_INPUTS  = 50,
    parameter N_NEURONS = 128,
    parameter BIT_WIDTH = 16    // Q8.8
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire [BIT_WIDTH-1:0]    I_in     [0:N_INPUTS-1],
    input  wire [BIT_WIDTH-1:0]    weights  [0:N_NEURONS*N_INPUTS-1],
    output wire                    spike_out[0:N_NEURONS-1],
    output wire [BIT_WIDTH-1:0]    V_out    [0:N_NEURONS-1]
);
```

### Timing

- **Latency**: 2 clock cycles (1 for multiply+MUX, 1 for LIF update)
- **Throughput**: 1 result per L clocks (L = bitstream length)
- **Clock rate**: 100-200 MHz on iCE40, up to 400 MHz on ECP5

## Synthesis flow

SC-NeuroCore uses Yosys (open-source) for synthesis:

```bash
# Step 1: Synthesise for iCE40
yosys -p "read_verilog hdl/sc_lif_neuron.v; synth_ice40 -top sc_lif_neuron; stat"

# Step 2: Place and route
nextpnr-ice40 --hx8k --json sc_lif_neuron.json --asc sc_lif_neuron.asc

# Step 3: Generate bitstream
icepack sc_lif_neuron.asc sc_lif_neuron.bin

# Step 4: Program FPGA
iceprog sc_lif_neuron.bin
```

For ECP5:
```bash
yosys -p "read_verilog hdl/sc_dense_layer_top.v; synth_ecp5 -top sc_dense_layer_top; stat"
nextpnr-ecp5 --85k --json sc_dense_layer_top.json --lpf constraints.lpf --textcfg sc_dense_layer_top.config
ecppack sc_dense_layer_top.config sc_dense_layer_top.bit
```

## Co-simulation verification

Before deploying to hardware, verify the Verilog against the Python
bit-true model using Verilator:

```bash
# Build testbench
verilator -Wall --cc hdl/sc_lif_neuron.v --exe tb_cosim.cpp --build

# Generate test vectors from Python
python -c "from sc_neurocore import FixedPointLIFNeuron; ..."

# Run and compare
./obj_dir/Vsc_lif_neuron
python compare_outputs.py  # cycle-by-cycle comparison
```

The Python `FixedPointLIFNeuron` is bit-exact to the Verilog. Every
intermediate value (multiply, shift, clamp) matches. If co-simulation
fails, the hardware will not reproduce the software.

See Tutorial 09 for the complete co-simulation flow.

## Q8.8 fixed-point design

All hardware arithmetic uses Q8.8 (16-bit signed, 8 fractional bits):

```
Multiply: (A × B) >> 8
  - A, B are int16 (Q8.8)
  - Product is int32
  - Right-shift by 8 realigns the binary point
  - Saturate to int16 range [-32768, 32767]

Add: A + B
  - Direct int16 addition
  - Overflow handled by saturation

Compare: A >= THRESHOLD
  - Simple signed comparison
  - THRESHOLD = 256 = 1.0 in Q8.8
```

### Overflow prevention

The LIF neuron's voltage can grow without bound under sustained
input. The hardware clamps voltage to the int16 range:

```verilog
// Saturation logic in sc_lif_neuron.v
wire signed [31:0] product = leak_k * V_mem;
wire signed [15:0] leak_term = (product > 32'sh7FFF_FF) ? 16'sh7FFF :
                                (product < -32'sh8000_00) ? -16'sh8000 :
                                product[23:8];
```

## LFSR random number generation

SC requires random bitstreams. The `sc_lfsr_16bit.v` module generates
pseudo-random sequences using a 16-bit linear feedback shift register:

```
Polynomial: x¹⁶ + x¹⁴ + x¹³ + x¹¹ + 1
Period: 65,535 (2¹⁶ - 1)
Taps: bits 15, 13, 12, 10
```

Each encoder needs its own LFSR with a different seed to ensure
independence. Correlated random numbers cause systematic errors in
SC arithmetic.

## Resource scaling

For an N-input, M-neuron network:

| Resource | Formula | 50→128 example |
|----------|---------|---------------|
| LUTs (AND gates) | N × M | 6,400 |
| LUTs (MUX trees) | M × log₂(N) | ~768 |
| LUTs (LIF neurons) | M × ~120 | ~15,360 |
| FFs (LIF state) | M × 48 | 6,144 |
| LFSRs | N + M | 178 |
| **Total LUTs** | | **~22,500** |

An iCE40 HX8K has 7,680 LUTs → fits a 20→32 layer.
An ECP5-85K has 83,640 LUTs → fits a 100→256 layer.

## Power estimation

SC power is dominated by switching activity (toggle rate ≈ 50%):

```
P = C_eff × V² × f × N_gates × activity

For iCE40 at 100 MHz, 1.2V:
  C_eff ≈ 5 fF per LUT
  Activity ≈ 0.5 (stochastic bitstreams toggle ~50%)
  N_gates = 22,500
  P ≈ 5e-15 × 1.44 × 1e8 × 22,500 × 0.5
  P ≈ 8.1 mW
```

Compare to a conventional 16-bit neural network on the same FPGA:
~50-100 mW. SC saves 6-12× on dynamic power.

## Design rules for SC hardware

1. **One LFSR per encoder** — shared LFSRs create correlation
2. **Minimum L=64 for meaningful computation** — below this, quantisation
   noise dominates (effective precision < 4 bits)
3. **L=256 is the sweet spot** — 8-bit effective precision, reasonable latency
4. **Clock gating on idle neurons** — neurons that haven't spiked recently
   can be clock-gated to save power
5. **Weight memory**: store Q8.8 weights in BRAM, not LUT-based registers,
   for networks larger than ~1K weights

## Further reading

- Tutorial 09: Hardware Co-simulation
- Tutorial 13: Fixed-Point Arithmetic
- Tutorial 14: Network Export & Deployment
- Hardware Guide: `docs/hardware/HARDWARE_GUIDE.md`
- FPGA Toolchain: `docs/hardware/FPGA_TOOLCHAIN_GUIDE.md`
