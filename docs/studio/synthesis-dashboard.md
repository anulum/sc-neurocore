# Synthesis Dashboard

The Synthesis Dashboard provides one-click FPGA synthesis from the Visual
SNN Studio. Generate Verilog from your neuron equations, then synthesise
to any supported FPGA target to see resource usage and timing estimates.

No other SNN framework offers visual FPGA synthesis from a web IDE.

## Quick Start

1. Write your ODE in the Equation Editor
2. Click **RTL** or **SV** to generate Verilog
3. Switch to the **FPGA** tab
4. Select your target FPGA (ice40, ECP5, Gowin, Xilinx)
5. Click **Synthesise**
6. View resource bars (LUTs, FFs, BRAMs, DSPs) and utilisation percentages

## Supported FPGA Targets

| Target | Device | Synth Tool | PnR Tool |
|--------|--------|-----------|----------|
| ice40 | iCE40 UP5K | Yosys `synth_ice40` | nextpnr-ice40 |
| ECP5 | LFE5U-25F | Yosys `synth_ecp5` | nextpnr-ecp5 |
| Gowin | GW1N | Yosys `synth_gowin` | (not supported) |
| Xilinx | Artix-7 | Yosys `synth_xilinx` | (not supported) |

## Resource Metrics

The dashboard shows four resource bars:

- **LUTs** — Look-Up Tables (combinational logic)
- **Flip-Flops** — Sequential elements (registers)
- **Block RAMs** — On-chip memory blocks
- **DSPs** — Digital Signal Processing blocks (multipliers)

Each bar shows absolute count and percentage utilisation against the
target device's capacity.

## Tool Installation

The synthesis pipeline uses open-source FPGA tools:

```bash
# macOS (Homebrew)
brew install yosys nextpnr

# Ubuntu/Debian
apt install yosys nextpnr-ice40

# Windows (MSYS2)
pacman -S mingw-w64-x86_64-yosys
```

The `/api/synth/tools-status` endpoint reports which tools are available.
The dashboard shows green/red indicators for each tool.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/synth/tools-status` | Detect installed EDA tools |
| POST | `/api/synth/run` | Verilog + target → Yosys synthesis |
| POST | `/api/synth/pnr` | JSON netlist → nextpnr place & route |

### POST /api/synth/run

```json
{
  "verilog": "module sc_lif(...); ... endmodule",
  "target": "ice40"
}
```

Returns resource counts, device capacity, utilisation percentages,
and a log excerpt from Yosys.

## End-to-End Workflow

```
ODE equation
  → [IR button] SC Intermediate Representation
  → [SV button] SystemVerilog
  → [FPGA tab → Synthesise] Yosys resource report
  → [PnR] nextpnr timing report (optional)
```

This complete pipeline — from differential equation to FPGA resource
estimate — runs in seconds from a single browser tab.
