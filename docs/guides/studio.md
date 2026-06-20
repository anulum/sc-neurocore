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
npm run test:e2e
```

The frontend capability shell is fail-closed for registered panels. If the
backend capability registry omits a panel contract, that panel is disabled
instead of assuming the underlying API or external tool is available.

The Playwright e2e suite starts the Vite dev server, mocks backend API
contracts at the browser boundary, and verifies the Admin operator status,
audit status, audit export, and worker-status workflows against the rendered
React application.

The Admin panel loads path-free audit health at startup and can request the
admin-gated audit export endpoint. Development-preview policy mode can still
use `X-Studio-Principal` plus the `studio.admin` role, but production
deployments should set `SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL=false` and
configure `SC_NEUROCORE_STUDIO_IDENTITY_FILE` with
`sc-neurocore.studio.identity.v1` service accounts. Store only SHA-256 token
hashes in that file, authenticate API calls with `Authorization: Bearer
<token>`, and give admin export accounts the `studio.admin` role.

Create the first local service account before enabling the production profile:

```bash
sc-neurocore studio-bootstrap-admin \
  --identity-file /etc/sc-neurocore/studio-identities.json \
  --principal-id svc-studio-admin
```

The bootstrap command prints the bearer token once and stores only its
SHA-256 digest in the identity file. Capture that token in the deployment
secret manager, set `SC_NEUROCORE_STUDIO_IDENTITY_FILE` to the written path,
and do not copy the token into repository files or shell history.

Administrators can inspect and update persistent service-account metadata from
the Admin panel Identity section or through
`/api/studio/identity/service-accounts`. The list and detail endpoints return
principal IDs, role lists, active state, and optional UTC expiry only; they do
not expose bearer-token hashes or filesystem paths. The PATCH endpoint updates
roles, active state, and expiry atomically, preserves the stored token hash,
reloads the backend authenticator after success, and records a dedicated
`studio.identity.service_account.update` audit event in addition to the normal
route-policy audit decision.

Interactive operators can use persistent browser users from the same identity
file. Add them offline through the maintained provisioning command:

```bash
printf '%s\n' "$STUDIO_OPERATOR_PASSWORD" | sc-neurocore studio-add-browser-user \
  --identity-file /etc/sc-neurocore/studio-identities.json \
  --username operator \
  --principal-id human-operator \
  --role studio.viewer \
  --password-stdin
```

The command writes username, principal ID, roles, active state, optional UTC
expiry, and a PBKDF2-HMAC-SHA256 password verifier while preserving existing
service accounts. The browser login form calls `POST /api/studio/auth/login`;
a successful login stores the returned bearer token in `sessionStorage`, sends
it as an `Authorization` header for subsequent API calls, and can revoke it
with `POST /api/studio/auth/logout`. Studio does not use auth cookies for this
session mode, so cookie CSRF tokens are intentionally not part of the current
contract. Set `SC_NEUROCORE_STUDIO_BROWSER_SESSION_TTL_SECONDS` to tune the
server-side session expiry; the default is 12 hours.

Repeated invalid browser-login attempts are throttled before another password
check is performed. The default policy permits five invalid attempts within
five minutes and then returns `429 browser_login_throttled` with `Retry-After`
for a 15-minute cooldown. Tune this deployment policy with
`SC_NEUROCORE_STUDIO_BROWSER_LOGIN_MAX_FAILURES`,
`SC_NEUROCORE_STUDIO_BROWSER_LOGIN_FAILURE_WINDOW_SECONDS`, and
`SC_NEUROCORE_STUDIO_BROWSER_LOGIN_COOLDOWN_SECONDS`. A successful login clears
prior invalid-attempt state for that username; disabled or expired users keep
their explicit failure reason and are not converted into throttle events.

Administrators can inspect and update browser-user lifecycle metadata from the
Admin panel Identity section or through `/api/studio/identity/browser-users`.
The Admin panel also creates browser users through the same route. Create,
detail, and PATCH responses return only username, principal ID, roles, active
state, and optional UTC expiry. Creation stores only a PBKDF2-HMAC-SHA256
verifier for the submitted password, reloads backend authentication
immediately, rejects duplicate usernames with `409`, and records
`studio.identity.browser_user.create` without password material. Role,
active-state, and expiry updates preserve the stored password verifier, reload
backend authentication immediately, and record both the route-policy audit
decision and a dedicated `studio.identity.browser_user.update` audit event.

Use `POST /api/studio/identity/browser-users/{username}/password` or the Admin
panel per-user secret field to rotate a browser user's password verifier. The
route preserves public metadata, writes a fresh PBKDF2-HMAC-SHA256 verifier,
reloads backend authentication, clears the login throttle bucket for that
username, revokes active browser sessions for the user's principal, and records
`studio.identity.browser_user.password.rotate` without password material.

The same Admin surface displays local worker health from
`/api/studio/jobs/status`. Configure `SC_NEUROCORE_STUDIO_JOB_ROOT` to keep
per-job working directories on an operator-selected disk, and tune
`SC_NEUROCORE_STUDIO_JOB_TIMEOUT_SECONDS` for cooperative worker timeouts. Use
`SC_NEUROCORE_STUDIO_JOB_MAX_ARTIFACT_BYTES` to cap each declared artifact
written by a worker. Use `SC_NEUROCORE_STUDIO_EDA_PROCESS_CPU_SECONDS` and
`SC_NEUROCORE_STUDIO_EDA_PROCESS_MEMORY_BYTES` to apply host-supported CPU and
memory ceilings to Yosys and nextpnr child processes. The status payload is
path-free and reports allowed job kinds plus active/completed/failed/timed-out
counts.

Training start, stop, and status routes now submit work through the same local
worker manager. The training monitor's SSE stream remains the live metric
channel, while the Admin queue records the bounded training job and its
path-free terminal artifact manifest.

Compile, synthesis, PnR, and full-pipeline routes also submit through the
bounded worker manager. Their HTTP responses remain synchronous for existing UI
flows, and the Admin queue records path-free artifacts at
`compiler/result.json`, `synthesis/result.json`,
`synthesis/multi-target-result.json`, `synthesis/pnr-result.json`, and
`pipeline/result.json`.

Job artifacts are served through the admin-gated
`/api/studio/jobs/{job_id}/artifacts/{artifact_path}` endpoint. The endpoint
only serves manifest-declared artifacts, revalidates the recorded size and
SHA-256 digest before returning bytes, and uses generic error details when an
artifact is missing or fails integrity checks.

The Admin panel queue uses `/api/studio/jobs` and `/api/studio/jobs/{job_id}`
for path-free job records. Those endpoints are admin-gated and expose status,
owner, request ID, timestamps, result metadata, and artifact manifests without
revealing the configured job-root path.

Studio also exposes a process-backed job-manager path for new backend work
that can be expressed as an importable `module:function` task with a
JSON-serializable payload and result. Process jobs use the same path-confined
artifact context and public manifest shape as thread-backed jobs, but they run
in a separate Python process so timeout or cancellation can terminate the
worker instead of leaving a long-running Python callable alive in the backend
process. Existing route closures remain on the thread-backed path until each
workflow is migrated to importable process tasks with explicit payload
contracts.

The Admin panel also uses the admin-gated `/api/studio/operator/status`
aggregate when available. That endpoint reports deployment profile,
route-policy enforcement, identity mode, audit health, worker health,
resource-limit posture, and capability counts without exposing local paths or
token material.

For production deployments, set
`SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE=production`. That profile fails closed
unless route policies are enforced, header principals are disabled, and the
identity file, audit log, and job root are all configured.

Studio ships deployment-profile packages for the three supported operator
contexts:

```bash
sc-neurocore studio-deployment-profile --studio-profile local
sc-neurocore studio-deployment-profile --studio-profile lab --output studio-lab-profile.json
sc-neurocore studio-deployment-profile --studio-profile server --format env --output studio-server.env
```

The package schema is `studio.deployment-profile.v1`. The `local` package keeps
the runtime profile in `development` with loopback-only hosts and origins for a
single workstation. The `lab` and `server` packages set the runtime profile to
`production`, require route-policy enforcement, disable header principals, and
include placeholders for the durable identity file, audit log, job root,
allowed hosts, allowed origins, preflight command, launch command, and backup
items. Package output contains placeholders and safe defaults only; bearer
tokens, password material, token hashes, and host-local paths must be supplied
outside the repository by the operator.

Before promoting a Studio deployment, run the release preflight from the same
environment that will launch the backend:

```bash
sc-neurocore studio-preflight --output studio-preflight.json
```

The command exits with status `0` only when the release posture passes. It
checks runtime settings, route-policy enforcement, disabled development header
principals, required admin route policies, a valid identity store with at least
one active unexpired `studio.admin` principal, audit-log readiness, and
job-root readiness. The JSON report uses schema `studio.preflight.v1` and is
safe for deployment logs: it reports booleans, counts, and stable check IDs
without local filesystem paths, bearer tokens, token hashes, passwords, or
password verifiers.

## Additional Panels (Blocks 2–6)

The Studio includes five additional panels beyond the core research workbench:

- **Compiler Inspector** — build SC IR, verify, emit SystemVerilog. [Details](../studio/compiler-inspector.md)
- **Synthesis Dashboard** — worker-backed Yosys synthesis for 4 FPGA targets, multi-target comparison, resource estimation. [Details](../studio/synthesis-dashboard.md)
- **Training Monitor** — live SNN training with 6 surrogate gradients, SSE metric streaming. [Details](../studio/training-monitor.md)
- **Network Canvas** — drag-and-drop populations and projections with React Flow, NIR export/import. [Details](../studio/network-canvas.md)
- **Integration** — worker-backed full pipeline (graph → compile → synthesise), project save/load. [Details](../studio/integration.md)

Full documentation: [Studio Hub](../studio/index.md)
