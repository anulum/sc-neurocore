# Synthesis Dashboard

The Synthesis Dashboard provides one-click FPGA synthesis from the Visual
SNN Studio. Generate Verilog from your neuron equations, then synthesise
to any supported FPGA target to see resource usage and timing estimates.

No other SNN framework offers visual FPGA synthesis from a web IDE.

## Quick Start

1. Write your ODE in the Equation Editor (switch to ODE mode)
2. Click **IR** to build the intermediate representation
3. Click **SV** to emit SystemVerilog
4. Switch to the **FPGA** tab
5. Select your target FPGA (ice40, ECP5, Gowin, Xilinx)
6. Click **Synthesise** for exact Yosys results, or **Estimate** for a quick heuristic

## Workflows

### Single-Target Synthesis

Select a target from the dropdown, click **Synthesise**. Yosys runs the
target-specific synthesis pass and returns resource counts with utilisation
bars.

### Multi-Target Comparison

Click **All Targets** to run Yosys synthesis on all four supported targets
simultaneously. Results appear in a comparison table showing LUTs, FFs,
BRAMs, and DSPs side-by-side, so you can pick the best target for your
design.

### Resource Estimation (No Yosys Required)

If Yosys is not installed, click **Estimate** after building the IR. The
estimator uses a heuristic based on IR operation count:

- Each IR op maps to ~2 LUTs + 1 FF
- LIF step op maps to ~12 LUTs + 8 FFs + 1 DSP

This gives a rough sizing before committing to a full synthesis run.

### End-to-End Pipeline

```
ODE equation
  → [IR button] SC Intermediate Representation
  → [SV button] SystemVerilog
  → [FPGA tab → Synthesise] Yosys resource report
  → [PnR] nextpnr timing report (ice40/ECP5 only)
```

From differential equation to FPGA resource estimate in seconds, from a
single browser tab.

## Supported FPGA Targets

| Target | Device | Synth Tool | PnR Tool | LUTs | FFs | BRAMs | DSPs |
|--------|--------|-----------|----------|------|-----|-------|------|
| ice40 | iCE40 UP5K | `synth_ice40` | nextpnr-ice40 | 5,280 | 5,280 | 30 | 0 |
| ECP5 | LFE5U-25F | `synth_ecp5` | nextpnr-ecp5 | 24,576 | 24,576 | 56 | 28 |
| Gowin | GW1N | `synth_gowin` | — | 20,736 | 20,736 | 41 | 0 |
| Xilinx | Artix-7 | `synth_xilinx` | — | 20,800 | 41,600 | 50 | 90 |

## Resource Metrics

The dashboard shows four resource bars:

- **LUTs** — Look-Up Tables (combinational logic)
- **Flip-Flops** — Sequential elements (registers)
- **Block RAMs** — On-chip memory blocks
- **DSPs** — Digital Signal Processing blocks (multipliers)

Each bar shows absolute count and percentage utilisation against the
target device's capacity. The multi-target comparison table shows all
four metrics across all targets.

## Tool Installation

The synthesis pipeline uses open-source FPGA tools:

```bash
# macOS (Homebrew)
brew install yosys nextpnr

# Ubuntu/Debian
apt install yosys nextpnr-ice40

# Windows (MSYS2)
pacman -S mingw-w64-x86_64-yosys

# Verify installation
yosys --version
nextpnr-ice40 --version
```

The `/api/synth/tools-status` endpoint reports which tools are available.
The dashboard shows green/grey indicators for each tool, with version
strings when available.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/synth/tools-status` | Detect installed EDA tools |
| POST | `/api/synth/run` | Verilog + target → Yosys synthesis |
| POST | `/api/synth/multi-target` | Verilog → all targets comparison |
| POST | `/api/synth/estimate` | IR op count → heuristic estimate |
| POST | `/api/synth/pnr` | JSON netlist → nextpnr place & route |

`/api/synth/run`, `/api/synth/multi-target`, and `/api/synth/pnr` execute
through the Studio local worker manager. They preserve the synchronous response
payloads shown below and additionally create Admin queue records with result
artifacts at `synthesis/result.json`,
`synthesis/multi-target-result.json`, and `synthesis/pnr-result.json`.

### GET /api/synth/tools-status

Returns availability and version for each tool:

```json
{
  "yosys": {"available": true, "version": "Yosys 0.40"},
  "nextpnr_ice40": {"available": true, "version": "nextpnr-ice40 0.7"},
  "nextpnr_ecp5": {"available": false, "version": null},
  "firtool": {"available": false, "version": null}
}
```

### POST /api/synth/run

```json
{
  "verilog": "module sc_lif(...); ... endmodule",
  "target": "ice40"
}
```

Returns:

```json
{
  "success": true,
  "target": "ice40",
  "resources": {"luts": 42, "ffs": 18, "brams": 0, "dsps": 0, "cells": 60, "wires": 85},
  "capacity": {"luts": 5280, "ffs": 5280, "brams": 30, "dsps": 0},
  "utilisation": {"luts": 0.8, "ffs": 0.3, "brams": 0.0, "dsps": 0.0},
  "log_excerpt": "..."
}
```

### POST /api/synth/multi-target

```json
{"verilog": "module sc_lif(...); ... endmodule"}
```

Returns synthesis results for all supported targets:

```json
{
  "targets": {
    "ice40": {"success": true, "target": "ice40", "resources": {...}, ...},
    "ecp5": {"success": true, "target": "ecp5", "resources": {...}, ...},
    "gowin": {"success": true, ...},
    "xilinx": {"success": true, ...}
  },
  "supported": ["ice40", "ecp5", "gowin", "xilinx"]
}
```

### POST /api/synth/estimate

Quick heuristic estimate without running Yosys:

```json
{"ir_op_count": 10, "target": "ice40"}
```

Returns:

```json
{
  "target": "ice40",
  "estimated": true,
  "resources": {"luts": 32, "ffs": 18, "brams": 0, "dsps": 1},
  "capacity": {"luts": 5280, "ffs": 5280, "brams": 30, "dsps": 0},
  "utilisation": {"luts": 0.6, "ffs": 0.3, "brams": 0.0, "dsps": 0.0}
}
```

### POST /api/synth/pnr

Place-and-route a Yosys JSON netlist (ice40 and ECP5 only):

```json
{"json_path": "/path/to/design.json", "target": "ice40"}
```

Returns timing analysis:

```json
{
  "success": true,
  "max_freq_mhz": 48.3,
  "critical_path": "clk -> neuron.v_reg -> spike_out",
  "log_excerpt": "..."
}
```
