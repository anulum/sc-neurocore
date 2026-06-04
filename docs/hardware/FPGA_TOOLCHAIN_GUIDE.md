# FPGA Toolchain & Hardware Guide

This document specifies every tool and board needed to synthesise, deploy,
and profile SC-NeuroCore on FPGA hardware.

---

## 1. Supported FPGA Families

### Xilinx / AMD

| Family | Example Part | Board | Approx. Cost | Notes |
|--------|-------------|-------|-------------|-------|
| Artix-7 | xc7a35t | Digilent Arty A7-35T | ~€130 | Best starter board. 33K LUTs, 100 MHz. |
| Artix-7 | xc7a100t | Digilent Nexys A7-100T | ~€270 | 63K LUTs, more BRAM for larger networks. |
| Zynq-7000 | xc7z020 | TUL PYNQ-Z2 | ~€120 | ARM Cortex-A9 + FPGA. Python-native I/O via PYNQ framework. |
| Zynq-7000 | xc7z020 | Digilent ZedBoard | ~€400 | Full-featured Zynq dev board. |
| Zynq UltraScale+ | xczu3eg | Avnet Ultra96-V2 | ~€300 | ARM Cortex-A53 + FPGA. Edge-AI target. |
| Kintex UltraScale | xcku040 | Xilinx KCU105 | ~€3,500 | High-performance target. |

### Intel / Altera

| Family | Example Part | Board | Approx. Cost | Notes |
|--------|-------------|-------|-------------|-------|
| Cyclone 10 LP | 10CL025YU256C8G | Terasic DE10-Lite | ~€80 | Budget option. 25K LE. |
| Cyclone V SoC | 5CSEMA5F31C6 | Terasic DE1-SoC | ~€300 | ARM Cortex-A9 + FPGA. |
| Arria 10 SoC | 10AS066 | Terasic DE10-Pro | ~€2,500 | High-end target. |

### Open-Source (Lattice / Gowin)

| Family | Board | Approx. Cost | Toolchain |
|--------|-------|-------------|-----------|
| Lattice iCE40 UP5K | iCEBreaker | ~€70 | Yosys + nextpnr (fully open) |
| Lattice ECP5 | OrangeCrab / ULX3S | ~€80–€150 | Yosys + nextpnr (fully open) |
| Gowin GW2A | Tang Primer 25K | ~€40 | Gowin IDE (free) or Yosys |

**Recommendation for first deployment:** PYNQ-Z2 (~€120). Python-native
MMIO access via the PYNQ framework means you can control the SC-NeuroCore
AXI-Lite register file directly from Jupyter notebooks.

---

## 2. Software Toolchains

### Xilinx Vivado (for Artix-7, Zynq, UltraScale)

| Item | Details |
|------|---------|
| **Download** | [AMD/Xilinx Downloads](https://www.xilinx.com/support/download.html) |
| **Release pin** | Vivado ML Standard `v2025.2` for current SHD / PYNQ-Z2 synthesis evidence |
| **Edition** | Vivado ML Standard (free for Artix-7 and Zynq-7000 parts) |
| **License** | Free for parts up to Artix-7 100T and Zynq-7020. Larger parts need paid license. |
| **Supported OS** | Ubuntu 20.04/22.04, RHEL 8/9, Windows 10/11 |
| **Disk space** | ~80 GB (full install), ~30 GB (device-limited install) |
| **Install** | Download unified installer → select "Vivado" → select target devices only |

```bash
# After install, source the setup script
source /tools/Xilinx/Vivado/2025.2/settings64.sh

# Verify
vivado -version

# Run SC-NeuroCore synthesis
vivado -mode batch -source deploy/fpga/vivado_synth.tcl
# or via the Python wrapper:
python tools/fpga_deploy.py --synth vivado --part xc7a35t
```

### Intel Quartus Prime (for Cyclone, Arria)

| Item | Details |
|------|---------|
| **Download** | [Intel FPGA Downloads](https://www.intel.com/content/www/us/en/products/details/fpga/development-tools/quartus-prime/resource.html) |
| **Edition** | Quartus Prime Lite (free for Cyclone 10, Cyclone V) |
| **License** | Free for Lite edition. Pro edition needed for Arria 10+. |
| **Supported OS** | Ubuntu 20.04/22.04, RHEL 8, Windows 10/11 |
| **Disk space** | ~30 GB (Lite), ~50 GB (Standard) |

```bash
# After install
export PATH=$PATH:/opt/intelFPGA_lite/24.1/quartus/bin

# Verify
quartus_sh --version

# Run SC-NeuroCore synthesis
python tools/fpga_deploy.py --synth quartus --part 10CL025YU256C8G
```

### Open-Source: Yosys + nextpnr (for iCE40, ECP5)

| Item | Details |
|------|---------|
| **Download** | [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build/releases) |
| **License** | Fully open source (ISC/MIT/GPL) |
| **Supported OS** | Linux, macOS, Windows (via MSYS2) |
| **Disk space** | ~1 GB |

```bash
# Pre-built binaries (recommended)
wget https://github.com/YosysHQ/oss-cad-suite-build/releases/latest/download/oss-cad-suite-linux-x64.tgz
tar xzf oss-cad-suite-linux-x64.tgz
source oss-cad-suite/environment

# Manual synthesis for iCE40
yosys -p "read_verilog hdl/sc_neurocore_top.v; synth_ice40 -top sc_neurocore_top -json build/sc.json"
nextpnr-ice40 --up5k --json build/sc.json --asc build/sc.asc --pcf deploy/fpga/icebreaker.pcf
icepack build/sc.asc build/sc.bin
```

### ASIC flow: Yosys + OpenROAD

The ASIC flow emits Yosys/OpenROAD-compatible decks, but OpenROAD is not
release-pinned yet. Treat OpenROAD area, timing, power, and GDSII numbers as
unpublished until the report records:

- the exact OpenROAD binary version or Docker image digest;
- the PDK name and revision;
- the Liberty, LEF, technology LEF, DRC, and LVS deck paths used for the run.

The current release can document Vivado `v2025.2` FPGA evidence; it must not
quote OpenROAD physical-design numbers without those pins.

### Co-Simulation: Icarus Verilog (all platforms, free)

| Item | Details |
|------|---------|
| **Download** | [Icarus Verilog](http://iverilog.icarus.com/) or `apt install iverilog` |
| **License** | GPL |
| **Purpose** | Behavioural simulation and Python co-sim verification |

```bash
# Run co-simulation (no FPGA needed)
iverilog -o tb_lif hdl/sc_lif_neuron.v hdl/tb_sc_lif_neuron.v
vvp tb_lif
python scripts/cosim_gen_and_check.py --check
```

### Formal RTL Checks: SymbiYosys + Yosys

| Item | Details |
|------|---------|
| **Download** | OSS CAD Suite or distro packages providing `sby`, `yosys`, and `cvc5` |
| **Purpose** | Discharge module-level safety and ordering properties before FPGA integration |
| **Current AER job** | `hdl/formal/sc_aer_priority_queue.sby` |

```bash
# Bounded proof when SymbiYosys and cvc5 are installed
sby -f hdl/formal/sc_aer_priority_queue.sby

# Parse/elaboration fallback used for local regression evidence
yosys -p "read_verilog -formal -sv hdl/sc_aer_priority_queue.v hdl/formal/sc_aer_priority_queue_formal.v; prep -top sc_aer_priority_queue_formal"
```

---

## 3. Hardware Requirements by Use Case

### Minimum: Co-Simulation Only (No FPGA Board)

- Any PC with Python 3.10+ and Icarus Verilog
- Verifies bit-exact parity between Python model and Verilog RTL
- Cost: **€0** (software only)

### Development: Synthesis + Basic Deployment

- **Board:** PYNQ-Z2 (~€120) or Arty A7-35T (~€130)
- **Cable:** Micro-USB (included with most boards)
- **Software:** Vivado ML Standard (free)
- **PC:** Any x86-64 with 16 GB RAM, 80 GB disk
- Cost: **~€120–€130**

### Energy Profiling (Issue #35)

- **Board:** PYNQ-Z2 or Arty A7 with current measurement shunt
- **Equipment:** Keysight N6705C power analyser (~€5,000) or Monsoon HVPM (~€800) or INA219 breakout board (~€5 on Adafruit)
- **Alternatively:** Vivado power estimation (post-implementation, no extra hardware)
- Cost: **€5–€800** (€5 for INA219 breakout + multimeter approach)

### Production / Edge Deployment

- **Board:** Ultra96-V2 (~€300) or custom carrier board
- **Tools:** Vivado + PetaLinux (free for ZU3EG)
- **Runtime:** PYNQ or bare-metal AXI-Lite driver

---

## 4. Deployment Flow

```
Python Network Spec
        │
        ▼
tools/fpga_deploy.py --emit-verilog
        │
        ▼
    hdl/*.v  (Verilog RTL)
        │
        ├──► Co-sim verification (Icarus Verilog, no FPGA needed)
        │
        ▼
tools/fpga_deploy.py --synth vivado --part xc7a35t
        │
        ▼
    Vivado project
        ├── utilization.rpt    (LUT/FF/BRAM usage)
        ├── timing.rpt         (setup/hold slack, Fmax)
        ├── power.rpt          (static + dynamic power estimate)
        └── *.bit              (FPGA bitstream)
        │
        ▼
    Program board via Vivado Hardware Manager or PYNQ overlay
```

---

## 5. Expected Resource Utilisation

Estimates for a single `sc_neurocore_top` instance (4 inputs, 1 LIF neuron,
256-step bitstream, Q8.8 fixed-point) on Artix-7:

| Resource | Estimated | Available (xc7a35t) | % |
|----------|----------:|--------------------:|--:|
| LUT | ~180 | 20,800 | 0.9% |
| FF | ~120 | 41,600 | 0.3% |
| BRAM (36Kb) | 0 | 50 | 0% |
| DSP48 | 0 | 90 | 0% |
| Fmax | ~200 MHz | — | — |

Scaling to a 16-input, 8-neuron dense layer with STDP: ~1,500 LUT, ~800 FF.

---

## 6. Procurement Links

| Item | Supplier | URL |
|------|----------|-----|
| Arty A7-35T | Digilent | https://digilent.com/shop/arty-a7/ |
| PYNQ-Z2 | TUL | https://www.tulembedded.com/FPGA/ProductsPYNQ-Z2.html |
| Nexys A7 | Digilent | https://digilent.com/shop/nexys-a7/ |
| Ultra96-V2 | Avnet | https://www.avnet.com/wps/portal/us/products/avnet-boards/avnet-board-families/ultra96-v2/ |
| DE10-Lite | Terasic | https://www.terasic.com.tw/cgi-bin/page/archive.pl?No=1021 |
| iCEBreaker | 1BitSquared | https://1bitsquared.com/products/icebreaker |
| INA219 | Adafruit | https://www.adafruit.com/product/904 |
| Vivado | AMD/Xilinx | https://www.xilinx.com/support/download.html |
| Quartus Lite | Intel | https://www.intel.com/content/www/us/en/products/details/fpga/development-tools/quartus-prime/resource.html |
| OSS CAD Suite | YosysHQ | https://github.com/YosysHQ/oss-cad-suite-build/releases |
| Icarus Verilog | Steve Williams | http://iverilog.icarus.com/ |

---

## 7. Academic / Volume Discounts

- **Xilinx University Program (XUP):** Free Vivado licenses + discounted boards for accredited institutions. Apply at https://www.xilinx.com/support/university.html
- **Intel FPGA University Program:** Free Quartus + device support. Apply at https://www.intel.com/content/www/us/en/developer/topic-technology/fpga-academic/overview.html
- **Digilent Academic:** 25–40% discount for verified educators/students at https://digilent.com/shop/academic/
