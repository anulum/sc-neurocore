<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->

# Deployment Guide

This guide covers deploying compiled SC-NeuroCore neurons to real FPGA
hardware: resource estimation, timing constraints, Cocotb testbenches,
project automation, CDC synchronisers, posit arithmetic, and bitstream
generation.

## Resource Estimation (Without Synthesis)

Estimate LUT/FF/DSP/BRAM usage in milliseconds — no Vivado/Quartus needed:

```python
from sc_neurocore.compiler.deployment import estimate_resources

# Generate Verilog first
verilog = neuron.to_verilog(module_name="sc_lif", data_width=16, fraction=8)

# Estimate resources
est = estimate_resources(verilog, data_width=16, has_dsp=True)
print(f"LUTs:  {est.luts}")
print(f"FFs:   {est.ffs}")
print(f"DSPs:  {est.dsps}")
print(f"BRAMs: {est.brams}")
print(f"Multipliers: {est.mul_count}")
print(f"Adders:      {est.add_count}")
print(f"Register bits: {est.reg_bits}")
```

### With vs Without DSP Blocks

```python
# Target with DSP (Artix-7) — multiplies map to DSP48E1
est_dsp = estimate_resources(verilog, has_dsp=True)
# → DSPs=3, LUTs=~120

# Target without DSP (QuickLogic EOS S3) — multiplies use LUTs
est_lut = estimate_resources(verilog, has_dsp=False)
# → DSPs=0, LUTs=~500
```

---

## Timing Constraints (SDC / XDC)

Generate timing constraint files matched to your target frequency:

### Xilinx (XDC)

```python
from sc_neurocore.compiler.deployment import generate_constraints

xdc = generate_constraints(
    "sc_lif",
    format="xdc",
    target_freq_mhz=100.0,  # 100 MHz → 10 ns period
)

with open("sc_lif.xdc", "w") as f:
    f.write(xdc)
```

Generated contents:
```tcl
create_clock -period 10.000 -name clk [get_ports clk]
set_input_delay -clock clk 2.000 [get_ports rst]
set_input_delay -clock clk 2.000 [get_ports {I_t[*]}]
set_output_delay -clock clk 2.000 [get_ports spike_out]
set_false_path -from [get_ports rst]
```

### Intel (SDC)

```python
sdc = generate_constraints("sc_lif", format="sdc", target_freq_mhz=450)
```

---

## Cocotb Testbenches

Generate Python-based verification testbenches (modern alternative to
Verilog testbenches):

```python
from sc_neurocore.compiler.deployment import generate_cocotb_testbench

tb = generate_cocotb_testbench(
    "sc_lif",
    data_width=16,
    fraction=8,
    n_steps=200,
    input_current=50.0,
)

with open("test_sc_lif.py", "w") as f:
    f.write(tb)
```

### Generated Tests

| Test | Verifies |
|------|----------|
| `test_sc_lif_spikes` | Neuron fires with constant current |
| `test_sc_lif_no_spike_zero_current` | No spikes with zero input |
| `test_sc_lif_reset_clears_state` | Reset returns to initial state |

### Running

```bash
# With Icarus Verilog
make SIM=icarus TOPLEVEL=sc_lif MODULE=test_sc_lif

# With Verilator
make SIM=verilator TOPLEVEL=sc_lif MODULE=test_sc_lif
```

---

## Project Automation (TCL)

### Xilinx Vivado

```python
from sc_neurocore.compiler.intelligence import generate_tcl_project

tcl = generate_tcl_project(
    "sc_lif",
    tool="vivado",
    part="xc7a35tcpg236-1",
    verilog_files=["sc_lif.v", "lfsr16.v"],
    constraint_file="sc_lif.xdc",
)

with open("build.tcl", "w") as f:
    f.write(tcl)
```

Run: `vivado -mode batch -source build.tcl`

Generated flow: `create_project` → `add_files` → `synth_design` →
`opt_design` → `place_design` → `route_design` → `write_bitstream` →
reports (utilisation, timing, power).

### Intel Quartus

```python
tcl = generate_tcl_project(
    "sc_lif",
    tool="quartus",
    part="5CSEMA5F31C6",
    constraint_file="sc_lif.sdc",
)
```

Run: `quartus_sh -t build.tcl`

---

## CDC Synchronisers (Multi-Clock Domain)

For designs with multiple clock domains (e.g., neuron at 100 MHz, bus at
50 MHz):

```python
from sc_neurocore.compiler.intelligence import generate_cdc_synchroniser

# Single-bit spike signal crossing
cdc = generate_cdc_synchroniser(
    "spike",
    width=1,
    stages=2,        # 2 for standard, 3 for higher MTBF
    src_clock="clk_neuron",
    dst_clock="clk_bus",
)
```

Generated Verilog uses `(* ASYNC_REG = "TRUE" *)` attributes for correct
placement in Xilinx FPGAs.

### Multi-Bit Signals

For multi-bit data (e.g., membrane voltage readback), use 3 stages or
consider a gray-code converter:

```python
cdc_data = generate_cdc_synchroniser("v_reg", width=16, stages=3)
```

---

## Posit Arithmetic

Posit numbers offer better dynamic range than fixed-point at the same bit
width. Useful for ultra-compact neurons (8-bit) and parameter transfer to
AI accelerators.

### Configurations

| Config | Bits | es | Max Value | Min Positive | Use Case |
|--------|:----:|:--:|:---------:|:------------:|----------|
| `POSIT8_0` | 8 | 0 | 64 | 1/64 | Compact neuron weights |
| `POSIT8_1` | 8 | 1 | 4096 | ~0.0002 | Wider range parameters |
| `POSIT16_1` | 16 | 1 | ~16M | ~6×10⁻⁸ | High-precision transfer |
| `POSIT16_2` | 16 | 2 | ~10¹⁸ | ~10⁻¹⁸ | Extreme dynamic range |

### API

```python
from sc_neurocore.compiler.intelligence import (
    POSIT8_0, posit_encode, posit_decode,
)

# Encode a membrane parameter
encoded = posit_encode(-65.0, POSIT8_0)  # → 8-bit posit integer
decoded = posit_decode(encoded, POSIT8_0)  # → ≈-65.0
```

---

## Open-Source Bitstream Flow (iCE40 / ECP5)

For Lattice iCE40 and ECP5 FPGAs, generate a complete Makefile using
open-source tools (Yosys + nextpnr):

### iCE40

```python
from sc_neurocore.compiler.intelligence import generate_oss_makefile

mk = generate_oss_makefile(
    "sc_lif",
    target="ice40",
    device="hx8k",
    package="ct256",
    freq_mhz=12.0,
    verilog_files=["sc_lif.v", "lfsr16.v"],
)

with open("Makefile", "w") as f:
    f.write(mk)
```

```bash
make         # → sc_lif.bin
make timing  # → timing report
make prog    # → program via iceprog
```

### ECP5

```python
mk = generate_oss_makefile(
    "sc_lif",
    target="ecp5",
    device="um5g-85k",
    package="CABGA381",
    freq_mhz=50.0,
)
```

### Tool Requirements

| Tool | Purpose | Install |
|------|---------|---------|
| `yosys` | Synthesis | `apt install yosys` |
| `nextpnr-ice40` | Place & Route (iCE40) | `apt install nextpnr` |
| `nextpnr-ecp5` | Place & Route (ECP5) | `apt install nextpnr` |
| `icepack` | Bitstream pack (iCE40) | `apt install fpga-icestorm` |
| `ecppack` | Bitstream pack (ECP5) | `apt install prjtrellis` |

---

## Complete Deployment Workflow

A typical deployment flow using all the tools:

```bash
# 1. Compile neuron to Verilog
python -c "
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from sc_neurocore.compiler.platforms import get_profile

profile = get_profile('artix7')
neuron = UniversalNeuron.from_schema('lif')
v = neuron.to_verilog(module_name='sc_lif',
    data_width=profile.data_width, fraction=profile.fraction)
open('sc_lif.v', 'w').write(v)
"

# 2. Generate AXI wrapper
python -c "
from sc_neurocore.hdl_gen.bus_interface import generate_bus_wrapper
v = generate_bus_wrapper('sc_lif', {'P_V_REST': 16}, bus='axi_lite')
open('sc_lif_axi_lite.v', 'w').write(v)
"

# 3. Generate constraints
python -c "
from sc_neurocore.compiler.deployment import generate_constraints
xdc = generate_constraints('sc_lif', format='xdc', target_freq_mhz=100)
open('sc_lif.xdc', 'w').write(xdc)
"

# 4. Generate Vivado project
python -c "
from sc_neurocore.compiler.intelligence import generate_tcl_project
tcl = generate_tcl_project('sc_lif_axi_lite', part='xc7a35tcpg236-1',
    verilog_files=['sc_lif.v', 'sc_lif_axi_lite.v'],
    constraint_file='sc_lif.xdc')
open('build.tcl', 'w').write(tcl)
"

# 5. Generate host driver
python -c "
from sc_neurocore.compiler.deployment import generate_host_driver
drv = generate_host_driver('sc_lif', {'P_V_REST': 16}, language='python')
open('sc_lif_driver.py', 'w').write(drv)
"

# 6. Build
vivado -mode batch -source build.tcl
```

Host-driver generation sanitizes module names and parameter names before
emitting Python or C source. Empty sanitized module names and sanitized
parameter-name collisions are rejected before any driver text is returned.

---

## Cross-References

- [Hardware Profiles Guide](hardware_profiles.md) — 65 platform profiles
- [Static Analysis Guide](static_analysis_guide.md) — guard bits, overflow proof, SVA
- [SoC Integration Guide](soc_integration_guide.md) — bus wrappers + drivers
- [Co-Simulation Guide](cosimulation_guide.md) — Python↔Verilog verification
- [Precision Modes Guide](precision_modes.md) — 11 Q-format modes
