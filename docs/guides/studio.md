# Visual SNN Design Studio

The Studio is a web-based research workbench for designing, simulating, and
analysing spiking neurons interactively. It combines 118 built-in neuron models,
custom ODE editing, 15 analysis views, and Verilog RTL generation in a single
interface.

## Installation

```bash
pip install sc-neurocore[studio]
```

This installs FastAPI and Uvicorn alongside the core package.

## Quick Start

```bash
sc-neurocore studio
```

Opens `http://127.0.0.1:8001` in your browser. The Studio starts in **Model
mode** with 118 neuron models browsable by category. Switch to **ODE mode** to
write custom equations.

To use a different port:

```bash
sc-neurocore studio --port 9000
```

## Two Modes

### Model Mode (118 models)

Browse all sc-neurocore neuron models grouped by category (Conductance, IF,
Oscillator, Bursting, Hardware, Network, Statistical, AI). Each model's
parameters appear as sliders. Models are auto-classified by firing pattern
(tonic, bursting, adapting, irregular, chaotic, silent) with colour-coded
badges.

**Pattern filter:** click a pattern badge in the model list to filter.

### ODE Mode (custom equations)

Write ODEs in Brian2-style syntax in the Monaco editor:

```
dv/dt = -(v - E_L) / tau_m + I / C

# threshold: v > -50
# reset: v = -65
```

Five built-in templates: LIF, Izhikevich, AdEx, Hodgkin-Huxley, FitzHugh-Nagumo.

## Analysis Views

The Studio provides 15 view tabs, each showing a different analysis of the
current simulation:

| View | Tab | What it shows |
|------|-----|---------------|
| Trace | Trace | Voltage + current + spike raster with zoom/pan |
| Phase portrait | Phase | v vs w trajectory with nullcline overlay |
| ISI histogram | ISI | Interspike interval distribution |
| f-I curve | f-I | Firing rate vs input current |
| Bifurcation | Bif | Parameter sweep → voltage attractor scatter |
| 2D Heatmap | 2D | Two-parameter sweep → firing rate colour map |
| Sensitivity | Sens | Parameter importance bar chart |
| Spike-triggered average | STA | Average voltage shape around spikes |
| Frequency response | Freq | Firing rate vs input frequency |
| Characterisation | Char | One-click dashboard: pattern, rheobase, f-I, sensitivity |
| Multi-model overlay | Multi | Compare 2-4 models in one plot |
| A/B Comparison | A/B | Split-view of two configurations |
| E-I Network | E-I | Excitatory-inhibitory network raster + population rates |
| Code generator | Code | Python script + one-liner for notebooks |
| Q8.8 Precision | Q8.8 | Float64 vs fixed-point comparison + error trace |
| Verilog RTL | RTL | Generated Verilog module from ODE (ODE mode only) |

### Trace View

The main view shows:

- **Voltage traces** for all state variables (colour-coded)
- **Current injection** subplot showing the input protocol
- **Spike raster** with red tick marks
- **Zoom/pan**: mouse wheel zooms the time axis centred on cursor, drag to pan,
  double-click to reset
- **Crosshair cursor** with tooltip showing exact time and voltage values
- **Axis labels**: mV for voltage, nA for current, ms for time
- **Imported data overlay**: paste CSV data to compare with simulation

### Characterisation Dashboard

Click **Char.** to run a one-click analysis that produces:

- Firing pattern classification
- Threshold current (rheobase estimate)
- f-I curve
- Top sensitive parameters
- State variable ranges

### E-I Network

Simulates a balanced excitatory-inhibitory LIF network. When the Rust engine
is installed, the entire simulation runs in compiled Rust — connectivity
construction, Poisson input generation, Euler integration, spike detection,
and rate binning happen in a single `py_simulate_ei_network()` call with zero
Python per-timestep overhead. Falls back to NumPy if the engine is unavailable.

Parameters (adjustable via sidebar sliders):

- Neuron counts (N exc, N inh)
- Synaptic weights (E→E, E→I, I→E, I→I)
- Connection probability
- External Poisson drive rate (Hz)

Displays a spike raster (blue = excitatory, red = inhibitory) and population
firing rate traces.

## Current Injection Protocols

Four injection protocols for all simulations:

| Protocol | Description |
|----------|-------------|
| Constant | Steady current for full duration |
| Step | 0 for first 20%, then I for remaining 80% |
| Ramp | Linear increase from 0 to I |
| Pulse train | 5ms on/off pulses at amplitude I |

## Interactive Features

- **Auto-simulate**: simulation reruns 250ms after any slider change
- **Keyboard shortcuts**: Space=run, 1-5=switch tabs, ?=help overlay
- **Session save/load**: save named sessions to localStorage
- **Shareable URLs**: state encoded in URL hash (click Share)
- **10 preset experiments**: threshold exploration, adaptation, bursting, chaos,
  hardware comparison, and more
- **CSV export**: download simulation data as comma-separated values
- **JSON export**: download raw simulation response
- **PNG export**: screenshot the current plot

## API Reference

The Studio backend exposes a REST API. All POST endpoints accept JSON.
Simulations are cached (LRU, 64 slots) for instant replay.

### Core Simulation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/simulate` | Run ODE simulation |
| POST | `/api/models/simulate` | Run named model simulation |
| POST | `/api/compare` | A/B comparison of two configs |
| POST | `/api/multi-simulate` | Simulate 2-4 models in parallel |
| POST | `/api/network/ei` | E-I balanced network simulation |

### Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/fi-curve` | Firing rate vs current sweep |
| POST | `/api/bifurcation` | Parameter sweep attractor map |
| POST | `/api/sensitivity` | Parameter sensitivity analysis |
| POST | `/api/heatmap` | Two-parameter firing rate heatmap |
| POST | `/api/characterize` | Full model characterisation |
| POST | `/api/freq-response` | Frequency response curve |
| POST | `/api/precision` | Float vs Q8.8 precision compare |
| POST | `/api/nullclines` | Nullcline computation for 2D ODEs |

The frequency-response endpoint runs the simulator with a true sinusoidal
current protocol for each frequency. The injected trace is
`I(t) = amplitude * sin(2*pi*frequency_hz*t)`, not a DC approximation.

### Resources

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/templates` | List ODE templates |
| GET | `/api/templates/{name}` | Get template by name |
| GET | `/api/models` | List all 118 models |
| GET | `/api/models/scan` | Classify all models by firing pattern |
| GET | `/api/models/{name}` | Get model detail (params, state vars) |
| GET | `/api/presets` | List preset experiments |
| GET | `/api/presets/{id}` | Get preset detail |
| POST | `/api/codegen` | Generate Python script |
| POST | `/api/compile` | Compile ODE to Verilog |
| GET | `/api/cache/stats` | Cache hit/miss statistics |
| GET | `/api/health` | Health check |

### Network Canvas

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/graph/models` | List neuron models available for graph populations |
| POST | `/api/graph/population` | Create a population node |
| POST | `/api/graph/projection` | Create a projection edge |
| POST | `/api/graph/validate` | Validate graph JSON and return structured errors |
| POST | `/api/graph/simulate` | Simulate a graph through the E-I backend |
| POST | `/api/graph/export-nir` | Export validated graph JSON to NIR-compatible JSON |
| POST | `/api/graph/import-nir` | Import NIR-compatible JSON to Studio graph JSON |

Graph and NIR import endpoints validate malformed JSON boundaries explicitly:
`populations` and `projections` must be lists, population IDs and NIR edge
endpoints must be non-empty strings, and numeric count/weight/probability
fields must be finite.

### Example: POST /api/simulate

```json
{
  "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
  "threshold": "v > -50",
  "reset": "v = -65",
  "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
  "init": {"v": -65.0},
  "dt": 0.1,
  "duration": 100.0,
  "current": 30.0
}
```

Response includes `time`, `states`, `spikes`, `spike_count`, `dt`, `n_steps`,
`stats` (rate_hz, isi_mean_ms, isi_cv, isi_histogram), `current_trace`, and
`pattern` (auto-classified firing behaviour).

## Development

```bash
# Terminal 1: backend
sc-neurocore studio --port 8001

# Terminal 2: frontend dev server (hot reload)
cd studio/frontend
npm install
npm run dev
```

The Vite dev server proxies `/api/*` to `http://127.0.0.1:8001`.

Production build:

```bash
cd studio/frontend
npm run build   # → studio/frontend/dist/
```

Frontend contract tests:

```bash
cd studio/frontend
npm test -- src/capabilityShell.test.ts
```

The frontend capability shell is fail-closed for registered panels. If the
backend capability registry omits a panel contract, that panel is disabled
instead of assuming the underlying API or external tool is available.

The Admin panel loads path-free audit health at startup and can request the
admin-gated audit export endpoint. Development-preview policy mode can still
use `X-Studio-Principal` plus the `studio.admin` role, but production
deployments should set `SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL=false` and
configure `SC_NEUROCORE_STUDIO_IDENTITY_FILE` with
`sc-neurocore.studio.identity.v1` service accounts. Store only SHA-256 token
hashes in that file, authenticate API calls with `Authorization: Bearer
<token>`, and give admin export accounts the `studio.admin` role.

## Additional Panels (Blocks 2–6)

The Studio includes five additional panels beyond the core research workbench:

- **Compiler Inspector** — build SC IR, verify, emit SystemVerilog. [Details](../studio/compiler-inspector.md)
- **Synthesis Dashboard** — Yosys synthesis for 4 FPGA targets, multi-target comparison, resource estimation. [Details](../studio/synthesis-dashboard.md)
- **Training Monitor** — live SNN training with 6 surrogate gradients, SSE metric streaming. [Details](../studio/training-monitor.md)
- **Network Canvas** — drag-and-drop populations and projections with React Flow, NIR export/import. [Details](../studio/network-canvas.md)
- **Integration** — full pipeline (graph → compile → synthesise), project save/load. [Details](../studio/integration.md)

Full documentation: [Studio Hub](../studio/index.md)
