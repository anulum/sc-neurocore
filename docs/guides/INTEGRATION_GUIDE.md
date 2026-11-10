<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Integration guide for the SHD accelerator on a PYNQ-Z2
-->

# Integration Guide: the SHD Accelerator on a PYNQ-Z2

This guide connects SC-NeuroCore's Spiking Heidelberg Digits (SHD) inference
core to a Xilinx PYNQ-Z2 board. It says what exists, how to build and drive
it, and what has not been validated yet.

## 1. Status

- **Co-simulation.** The SHD core `hdl/sc_shd_top.v` is a three-stage pipeline
  (axonal delay, dense layer, Vmin LIF). Icarus co-simulation
  (`hdl/tb_sc_shd_top.v`) matches its bit-true Q8.8 reference
  (`tools/shd_q88_reference.py`).
- **Synthesis.** The core is synthesised out of context for the Zynq XC7Z020 at
  100 MHz. A bitstream was written on 2026-04-13. Both runs were made without
  weights or delays.
- **No trained model in hardware.** The delay tables and the weight ROM are
  loaded with `$readmemh` only in simulation. The AXI-Lite wrapper has no
  weight or delay register. A bitstream built with the script below therefore
  runs an empty model. A path to load weights and delays is still open.
- **No board run.** The driver and the demo below have not been run on a
  PYNQ-Z2. No accuracy, latency or power figure exists for the board.
- **Model accuracy.** The trained floating-point model reaches 75.2 % on the
  SHD test set. That figure is for the trained model, not for the hardware.

These statements come from `ARCHITECTURE.md`, section "SHD FPGA Pipeline". It
also gives the synthesis figures and the report files that bind them.

## 2. Components

| File | Role |
|------|------|
| `hdl/sc_shd_top.v` | SHD inference core |
| `hdl/sc_shd_axi_wrapper.v` | AXI4-Lite slave that the Zynq processing system drives |
| `hdl/pynq/create_block_design.tcl` | Vivado project, block design, implementation and bitstream |
| `hdl/constraints/pynq_z2.xdc` | PYNQ-Z2 constraints |
| `hdl/pynq/sc_shd_driver.py` | `SHDAccelerator`, the Python driver |
| `hdl/pynq/demo_shd_fpga.py` | Demo script for the board |
| `data/masquelier_shd/fpga_artifacts/dcls_max/` | Trained int8 weights, integer delays and Q16.16 scales (`scales.json`) |

## 3. Build the bitstream

Run Vivado in batch mode from the directory that should hold the project.
The script creates `./vivado_project`:

```bash
vivado -mode batch -source hdl/pynq/create_block_design.tcl
```

The script does the following:

- targets part `xc7z020clg400-1`;
- adds `sc_vmin_lif_neuron.v`, `sc_axonal_delay.v`, `sc_dense_int8_sparse.v`,
  `sc_shd_top.v` and `sc_shd_axi_wrapper.v` with the PYNQ-Z2 constraints;
- sets the processing-system clock `FCLK_CLK0` to 100 MHz;
- connects the wrapper, as block-design instance `sc_shd_0`, to `M_AXI_GP0`
  through an AXI interconnect;
- runs synthesis, implementation and `write_bitstream`.

The bitstream is `vivado_project/sc_shd_pynq.runs/impl_1/system_wrapper.bit`.
The hardware platform is `vivado_project/sc_shd_pynq.xsa`. PYNQ loads a
bitstream together with the block design's `.hwh` file of the same base name,
so copy both to the board, for example as `sc_shd.bit` and `sc_shd.hwh`.

Because of section 1, this bitstream carries no trained model.

## 4. Drive the core from Python on the board

```python
from pynq import Overlay
from sc_shd_driver import SHDAccelerator

overlay = Overlay("sc_shd.bit")  # with sc_shd.hwh in the same directory
accel = SHDAccelerator(overlay.sc_shd_0)
accel.set_scales(scale_l1, scale_l2, scale_l3)  # signed Q16.16 values from scales.json
predicted = accel.classify(spike_raster)        # binary array of shape (T, 140)
```

The block-design script names the instance `sc_shd_0`, and `overlay.ip_dict`
lists the instance names of a loaded overlay. `classify` does the following:

1. writes the input length;
2. starts the core;
3. writes each input step as five 32-bit words and a commit;
4. polls the done bit;
5. returns the argmax of the 20 output sums.

After the last input, the core runs 30 zero-padding steps and three
pipeline-drain clocks by itself. It ignores further commits.

Register map, from the header of `hdl/sc_shd_axi_wrapper.v`:

| Offset | Access | Register |
|--------|--------|----------|
| `0x00` | R/W | `CTRL`: bit 0 start (write), bit 1 busy, bit 2 done (held until the next start) |
| `0x04` | R/W | `T_ORIG`: input length, 1 to 65,503 steps |
| `0x08`, `0x0C`, `0x10` | R/W | `SCALE_L1`, `SCALE_L2`, `SCALE_L3`: Q16.16 per-layer scales |
| `0x14` to `0x24` | W | `SPIKE_IN_0` to `SPIKE_IN_4`: the 140 input bits of one step |
| `0x28` | W | `SPIKE_COMMIT`: take the input bits and advance one step |
| `0x40` to `0x8C` | R | `OUT_V_0` to `OUT_V_19`: signed 32-bit output sum of each class |

The wrapper answers with SLVERR and changes nothing in four cases: a start
while busy, an invalid length, an unaligned write, and a write to a reserved
or read-only address.

Each input step costs six MMIO writes from Python. The time per sample is
therefore set by the Python loop, not by the 100 MHz core clock. No latency
has been measured.

## 5. The package driver `sc_neurocore.drivers`

`sc_neurocore.drivers.SC_NeuroCore_Driver` does not drive the SHD core.

- In `HARDWARE` mode it expects an overlay with an IP named `scpn_layer_1_0`,
  which no design in this repository provides. Its `run_step` raises
  `NotImplementedError`.
- In `EMULATION` mode, `run_step` returns a seeded pseudo-random vector of 16
  values for software development.

See [Hardware Drivers](../api/drivers.md).

## 6. Before a first board run

1. Add a path that loads the trained weights and delays into the hardware,
   through registers or through memory initialised in the bitstream.
2. Build the bitstream with the model and keep its Vivado reports.
3. Run the demo on the board. Compare the 20 output sums with the Q8.8
   reference for the same inputs.
4. Quote a board accuracy, latency or power figure only after that run.
