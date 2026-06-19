# Visual SNN Design Studio

> **Status:** Development preview — functional but under active development.
> API and UI may change between releases until the v4.0 stable API freeze.

The Visual SNN Design Studio is a web-based IDE for the complete spiking
neural network lifecycle: design neuron models, build networks, train with
surrogate gradients, compile to SystemVerilog, and synthesise to FPGA — all
from a single browser tab.

## First-Time Onboarding

On first visit, an 8-step guided tour introduces the key features:
model browser, ODE mode, analysis views, FPGA pipeline, network canvas,
training monitor, and keyboard shortcuts. The tour is dismissable and
won't appear again (stored in localStorage).

## Launch

```bash
pip install sc-neurocore[studio]
sc-neurocore studio                # http://127.0.0.1:8001
sc-neurocore studio --port 9000    # custom port
```

For development (hot reload):

```bash
# Terminal 1: backend
py -3.12 -c "from sc_neurocore.studio.app import create_app; import uvicorn; uvicorn.run(create_app(), host='127.0.0.1', port=8001)"

# Terminal 2: frontend (Vite dev server with HMR)
cd studio/frontend && npm run dev  # http://localhost:5173
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Visual SNN Design Studio                       │
├──────────┬──────────┬──────────┬──────────┬──────────────────────┤
│ Equation │ Network  │ Training │ Compiler │ Synthesis            │
│ Editor   │ Canvas   │ Monitor  │ Inspector│ Dashboard            │
├──────────┴──────────┴──────────┴──────────┴──────────────────────┤
│              React + TypeScript + Zustand + React Flow            │
├──────────────────────────────────────────────────────────────────┤
│                    FastAPI Backend (Python)                       │
├──────────┬──────────┬──────────┬──────────┬──────────────────────┤
│ 118      │ PyTorch  │ equation │ Yosys    │ Project              │
│ neurons  │ training │ compiler │ nextpnr  │ save/load            │
├──────────┴──────────┴──────────┴──────────┴──────────────────────┤
│              SC-NeuroCore Python + Rust Engine                    │
└──────────────────────────────────────────────────────────────────┘
```

## Platform Contracts

The backend exposes typed platform contracts before UI panels call into
runtime features:

- `CapabilityRegistry` lists Studio capabilities, user-visible status,
  requirement health, evidence class, UI placement, and documentation path.
- `/api/studio/capabilities` and `/api/studio/capabilities/{capability_id}`
  return non-secret capability health payloads for the frontend shell.
- `PolicyGateway` is the fail-closed route authorization contract used for
  protected Studio API surfaces. Public routes may run without a principal;
  authenticated and admin routes require an explicit policy and emit audit
  decisions.

The policy gateway is intentionally a platform contract first. Existing public
health and capability routes stay backward compatible while protected route
integration is added panel by panel.

## Panels

### Equation Editor & Model Browser

Browse 118 neuron models by category (integrate-and-fire, biophysical,
stochastic, hardware emulators, AI-optimised). Select a model and
adjust parameters with sliders — the trace view updates live.

Switch to ODE mode to write custom equations with syntax highlighting:

```
dv/dt = -(v - E_L) / tau_m + I / C
```

The Monaco editor provides SC-NeuroCore-specific syntax highlighting:
- **Blue bold:** ODE derivatives (`dv/dt`, `dw/dt`)
- **Teal:** parameters (`E_L`, `tau_m`, `g_Na`, `V_threshold`)
- **Light blue:** state variables (`v`, `w`, `m`, `h`, `n`)
- **Yellow:** functions (`exp`, `sqrt`, `tanh`)
- **Purple:** directives (`threshold`, `reset`)
- **Green:** comments (`# ...`)

### 18+ Analysis Views

| View | Description |
|------|-------------|
| Trace | Membrane voltage + spike raster + current protocol |
| Phase | Phase portrait with nullclines (2D ODE) |
| ISI | Inter-spike interval histogram |
| f-I | Firing rate vs. injected current curve |
| Bifurcation | Parameter sweep → attractor diagram |
| 2D Heatmap | Two-parameter sweep → firing rate heatmap |
| Sensitivity | One-at-a-time parameter sensitivity |
| STA | Spike-triggered average |
| Frequency | Frequency response (sinusoidal input) |
| Characterise | One-click dashboard: pattern + f-I + sensitivities |
| Multi-model | Overlay up to 4 models for comparison |
| A/B Compare | Side-by-side model comparison |
| E-I Network | Balanced excitatory-inhibitory network raster + rates |
| Code | Python script generator + clipboard one-liner |
| Q8.8 | Float vs. fixed-point co-simulation diff |
| RTL | Equation → Verilog compiler output |
| IR | SC Intermediate Representation viewer |
| FPGA | Synthesis resource bars |
| Train | Live training monitor |
| Canvas | Network graph editor |

### Data Export

| Format | Contents | Use Case |
|--------|----------|----------|
| SVG | Vector traces with axes, legend, spike markers | Paper figures, LaTeX, Inkscape |
| CSV | Time column + state variables | External analysis (MATLAB, R, pandas) |
| JSON | Full simulation result (time, states, spikes, params) | Reproducibility, archival |

The SVG export produces a dark-themed 800x400 plot with grid lines,
axis labels, colour-coded state variables, spike markers, and a legend.
Polylines are downsampled to 2000 points for file-size efficiency
while preserving trace shape.

### Network Canvas

Drag-and-drop populations (excitatory = blue, inhibitory = red) and
connect them with projections by dragging between node handles.
Configure weights, delays, and connection probability per projection.

Export/import networks in [NIR](https://neuroir.org/) format for
interoperability with snnTorch, Norse, and SpikingJelly.

### Training Monitor

Start SNN training from the browser. Configure:
- Dataset (synthetic 64D or MNIST 784D)
- 6 surrogate gradient functions (atan, fast sigmoid, superspike, sigmoid, STE, triangular)
- Learnable beta (membrane leak) and threshold
- Batch size, learning rate, timesteps, epochs

Watch loss curves, accuracy, per-layer spike rates, and parameter
evolution update in real time via Server-Sent Events.

### Compiler Inspector

Build SC Intermediate Representation from ODE equations, verify the
graph, and emit synthesisable SystemVerilog. View the IR text and
generated Verilog side-by-side with a verification badge.

### Synthesis Dashboard

Run Yosys synthesis on generated Verilog for 4 FPGA targets:

| Target | Device | LUTs | FFs | BRAMs | DSPs |
|--------|--------|------|-----|-------|------|
| ice40 | iCE40 UP5K | 5,280 | 5,280 | 30 | 0 |
| ECP5 | LFE5U-25F | 24,576 | 24,576 | 56 | 28 |
| Gowin | GW1N | 20,736 | 20,736 | 41 | 0 |
| Xilinx | Artix-7 | 20,800 | 41,600 | 50 | 90 |

Multi-target comparison table shows resource usage across all targets.
Quick heuristic estimation available without Yosys installed.

### Full Pipeline

One-click pipeline from the Network Canvas:

```
Network Graph → Validate → Simulate → Compile → Synthesise
```

### Project Save/Load

Save complete workspace state (equations, parameters, network graph,
synthesis target, training config) as JSON files. Restore any saved
project from the sidebar.

## Detailed Guides

- [Synthesis Dashboard](synthesis-dashboard.md) — FPGA targets, multi-target comparison, API
- [Training Monitor](training-monitor.md) — surrogates, cell types, SSE streaming, API
- [Network Canvas](network-canvas.md) — populations, projections, NIR format, API
- [Compiler Inspector](compiler-inspector.md) — IR build/verify/emit, co-simulation, API
- [Integration & Projects](integration.md) — pipeline, save/load, API

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | React 19 + TypeScript (strict) |
| Graph editor | @xyflow/react (React Flow) |
| State | Zustand |
| Responsive | 4 breakpoints (1024, 768, 480px) |
| Build | Vite |
| Backend | FastAPI |
| Simulation | SC-NeuroCore Python + Rust engine |
| Training | PyTorch (optional `[research]` extra) |
| Synthesis | Yosys + nextpnr (external, optional) |

## API Reference

The Studio exposes 40+ REST endpoints. See the per-block documentation
for complete API details with request/response examples.

| Prefix | Block | Endpoints |
|--------|-------|-----------|
| `/api/simulate`, `/api/models/*` | Phase 1 | ODE/model simulation, templates, presets |
| `/api/fi-curve`, `/api/bifurcation`, `/api/heatmap`, ... | Phase 1 | Analysis views |
| `/api/ir/*` | Compiler Inspector | IR build, verify, emit SV, co-sim |
| `/api/synth/*` | Synthesis Dashboard | Yosys synthesis, multi-target, estimate |
| `/api/training/*` | Training Monitor | Start/stop, SSE stream, surrogates |
| `/api/graph/*` | Network Canvas | Populations, projections, validate, simulate, NIR |
| `/api/project/*`, `/api/pipeline/*` | Integration | Save/load, full pipeline |

## Requirements

```bash
pip install sc-neurocore[studio]   # FastAPI, uvicorn, React frontend
pip install sc-neurocore[research] # PyTorch training (optional)
# Optional Rust engine: use a matching release wheel or build the local bridge.
cd bridge && maturin develop --release
# Yosys + nextpnr for FPGA synthesis (optional)
```

### Rust Engine

The optional `sc_neurocore_engine` bridge provides SIMD-accelerated
simulation. Use a matching release wheel when one is provided for your
platform, or build it from the source checkout with `maturin develop --release`.
When installed, the Studio automatically uses it for E-I network simulation and
batch model runs. Without it, NumPy fallbacks are used — everything works, just
slower for large networks.

---

<p align="center">
  <a href="https://www.anulum.li">
    <img src="assets/anulum_logo_company.jpg" width="180" alt="ANULUM">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.anulum.li">
    <img src="assets/fortis_studio_logo.jpg" width="180" alt="Fortis Studio">
  </a>
  <br>
  <em>Developed by <a href="https://www.anulum.li">ANULUM</a> / Fortis Studio</em>
</p>
