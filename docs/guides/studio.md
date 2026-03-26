# Visual SNN Design Studio

The Studio is a web-based Equation Playground for designing and simulating
spiking neurons interactively. Write ODE equations, adjust parameters with
sliders, and see voltage traces and spike rasters update in real time.

## Installation

```bash
pip install sc-neurocore[studio]
```

This installs FastAPI and Uvicorn alongside the core package.

## Quick Start

```bash
sc-neurocore studio
```

This starts the backend server on `http://127.0.0.1:8001` and opens your
browser. Select a neuron template from the dropdown, adjust parameters,
and click **Simulate**.

To use a different port:

```bash
sc-neurocore studio --port 9000
```

## Interface

### Template Dropdown

Five built-in neuron models:

| Template | Variables | Description |
|----------|-----------|-------------|
| LIF | v | Leaky integrate-and-fire with threshold and reset |
| Izhikevich | v, u | Regular spiking with recovery variable |
| AdEx | v, w | Adaptive exponential with subthreshold oscillations |
| Hodgkin-Huxley | v, m, h, n | 4-variable conductance model with Na/K channels |
| FitzHugh-Nagumo | v, w | 2-variable relaxation oscillator |

Selecting a template loads its equations, parameters, threshold, reset rule,
and default current into the editor and sliders.

### Equation Editor

A Monaco (VS Code) editor where you write ODE equations in Brian2-style
syntax:

```
dv/dt = -(v - E_L) / tau_m + I / C

# threshold: v > -50
# reset: v = -65
```

Equations must follow the `d<var>/dt = <expression>` format.
Threshold and reset lines start with `# threshold:` and `# reset:`.

Supported functions in expressions: `exp`, `log`, `sqrt`, `abs`, `sin`,
`cos`, `tanh`, `sigmoid`, `clip`, `max`, `min`. Constants: `pi`.
Multi-variable reset uses semicolons: `v = -65; w = w + 0.08`.

### Parameter Sliders

Each parameter from the ODE equations gets an auto-ranging slider.
The simulation controls below the parameters set:

- **I** — input current (nA)
- **dt** — integration timestep (ms)
- **T** — simulation duration (ms)

### Voltage Plot

A dark-themed canvas showing:

- **Voltage trace** (blue line) — membrane potential over time
- **Additional state variables** (green, orange, red) for multi-variable models
- **Spike raster** (red vertical bars) below the trace
- **Spike count** in the bottom-left corner

## API Reference

The Studio backend exposes a REST API for programmatic use.

### POST /api/simulate

Run an ODE simulation.

**Request body:**

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

**Response:**

```json
{
  "time": [0.0, 0.1, 0.2, ...],
  "states": {"v": [-65.0, -64.5, ...]},
  "spikes": [142, 287, 431],
  "spike_count": 3,
  "dt": 0.1,
  "n_steps": 1000
}
```

Simulations are capped at 100,000 steps. Traces longer than 5,000 points
are downsampled for browser performance.

### GET /api/templates

Returns a list of all neuron templates with their default parameters.

### GET /api/templates/{name}

Returns a single template by name (`lif`, `izhikevich`, `adex`,
`hodgkin_huxley`, `fitzhugh_nagumo`).

### GET /api/health

Returns `{"status": "ok"}`.

## Development

To work on the frontend:

```bash
# Terminal 1: start backend
sc-neurocore studio --port 8001

# Terminal 2: start Vite dev server (hot reload)
cd studio/frontend
npm install
npm run dev
```

The Vite dev server proxies `/api/*` requests to the backend at port 8001.

To build the frontend for production:

```bash
cd studio/frontend
npm run build
```

Output goes to `studio/frontend/dist/`.

## Roadmap

Phase 1 (current) delivers the Equation Playground. Future phases:

- **Phase 2:** Network canvas with drag-and-drop populations and projections
- **Phase 3:** Live training monitor with surrogate gradient selection
- **Phase 4:** Compiler inspector showing IR and generated Verilog
- **Phase 5:** One-click FPGA synthesis dashboard with resource charts
