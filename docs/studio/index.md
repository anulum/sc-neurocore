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
- The frontend shell loads the capability registry during startup, surfaces
  aggregate capability health in the header, lists each capability with status,
  evidence, missing requirements, and documentation links, and disables
  registered panels and matching toolbar/keyboard activation paths when backend
  or external-tool requirements are unavailable.
- The default registry covers the stateful Studio panel families: simulation
  workbench, analysis suite, compiler inspector, synthesis dashboard, training
  monitor, network canvas, project workspace, and export tools. Missing
  registry entries fail closed in the frontend shell.
- `PolicyGateway` is the fail-closed route authorization contract used for
  protected Studio API surfaces. Public routes may run without a principal;
  authenticated and admin routes require an explicit policy and emit audit
  decisions.
- Runtime route-policy enforcement is opt-in for the development preview via
  `SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES=true`. When enabled, protected
  HTTP routes require an authenticated principal. Development builds may still
  use `X-Studio-Principal` plus comma-separated `X-Studio-Roles`; production
  deployments should set `SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL=false`.
- Durable service-account identity is configured with
  `SC_NEUROCORE_STUDIO_IDENTITY_FILE`. The file uses
  `sc-neurocore.studio.identity.v1`, stores SHA-256 bearer-token hashes rather
  than raw tokens, grants explicit roles, and can include UTC expiry timestamps.
  Requests authenticate with `Authorization: Bearer <token>`; invalid,
  disabled, or expired tokens fail closed and emit distinct audit reasons.
- The same identity file can include persistent `browser_users` with
  PBKDF2-HMAC-SHA256 password verifiers, explicit roles, active state, and
  optional UTC expiry timestamps. Browser users authenticate through
  `POST /api/studio/auth/login`, which returns an expiring bearer token for
  `Authorization: Bearer <token>` requests. Studio does not set browser auth
  cookies in this mode, so cookie CSRF tokens are not part of the current
  contract; deployments should protect the browser against script injection
  and transport the bearer token only over HTTPS or loopback-local channels.
- Browser bearer sessions are server-side and expire after
  `SC_NEUROCORE_STUDIO_BROWSER_SESSION_TTL_SECONDS`, defaulting to 12 hours.
  `/api/studio/auth/session` returns the current principal without token
  material, and `POST /api/studio/auth/logout` revokes the presented session.
- Browser login attempts are guarded by an in-memory, per-username lockout
  policy. Defaults allow five invalid password attempts within five minutes
  and then return `429 browser_login_throttled` with a `Retry-After` header for
  a 15-minute cooldown. Tune the policy with
  `SC_NEUROCORE_STUDIO_BROWSER_LOGIN_MAX_FAILURES`,
  `SC_NEUROCORE_STUDIO_BROWSER_LOGIN_FAILURE_WINDOW_SECONDS`, and
  `SC_NEUROCORE_STUDIO_BROWSER_LOGIN_COOLDOWN_SECONDS`. Throttle denials emit
  `studio.auth.login` audit rows without password material.
- First-deployment service-account identity files are created offline with
  `sc-neurocore studio-bootstrap-admin --identity-file <path>`. The command
  writes only the SHA-256 token hash to disk, returns the bearer token once to
  the operator, refuses to overwrite existing files unless `--allow-overwrite`
  is supplied, and applies owner-only file permissions where the host platform
  supports POSIX modes.
- Persistent browser users are added offline with
  `sc-neurocore studio-add-browser-user --identity-file <path> --username <name>
  --principal-id <principal> --role <role> --password-stdin`. The command reads
  the password from standard input, writes only the PBKDF2-HMAC-SHA256 verifier
  to the same identity file, preserves existing service accounts, and returns a
  password-free JSON summary.
- `/api/studio/identity/service-accounts` returns an admin-only, token-free
  service-account inventory for the configured persistent identity file.
  `/api/studio/identity/service-accounts/{principal_id}` returns one account,
  and `PATCH /api/studio/identity/service-accounts/{principal_id}` updates
  roles, active state, and optional UTC expiry while preserving the stored
  bearer-token hash. The backend reloads the identity authenticator after a
  successful update, so role changes apply without a Studio restart. Updates
  fail with `409` if they would remove the final active unexpired
  `studio.admin` principal.
- The Admin panel includes an Identity section backed by the same endpoints.
  It displays principal IDs, active state, expiry, and role lists without
  exposing token hashes or local identity-file paths. Role changes emit both
  the route-policy audit decision and a dedicated
  `studio.identity.service_account.update` audit event.
- `/api/studio/identity/browser-users` returns an admin-only, password-free
  browser-user inventory for the same identity file. The payload includes
  usernames, principal IDs, roles, active state, and expiry only.
  `POST /api/studio/identity/browser-users` creates a new browser user,
  stores only a PBKDF2-HMAC-SHA256 verifier for the supplied password,
  reloads authentication immediately, and emits
  `studio.identity.browser_user.create` without password material. Duplicate
  usernames fail closed with `409`.
  `/api/studio/identity/browser-users/{username}` returns one password-free
  browser-user record, and `PATCH /api/studio/identity/browser-users/{username}`
  updates roles, active state, and optional UTC expiry while preserving the
  stored password verifier. The backend reloads the identity authenticator
  after a successful update, so disabling a browser user or changing roles
  applies without a Studio restart. The same last-admin guard prevents
  browser-user changes that would remove the final active admin path. The Admin
  panel exposes create, lifecycle, and secret-rotation controls without
  returning password verifier material.
- Admins rotate a browser user's password with
  `POST /api/studio/identity/browser-users/{username}/password`. The request
  writes a fresh PBKDF2-HMAC-SHA256 verifier, preserves password-free user
  metadata, reloads authentication immediately, clears that user's login
  throttle bucket, revokes active browser sessions for the user's principal,
  and emits `studio.identity.browser_user.password.rotate` without password
  material. The Admin panel exposes this as a per-user secret-rotation control.
- Policy decisions can be persisted to an append-only JSONL audit log by
  setting `SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH` to a writable file path. Each
  `studio.audit.v1` line records the UTC timestamp, policy action, route
  template, principal identifier when present, request correlation ID when
  available, allow/deny decision, decision reason, previous event hash, and a
  SHA-256 event hash over the canonical row content.
- If a protected route cannot append its required audit event, Studio returns
  `503 audit_append_failed` instead of executing the operation without audit
  evidence. `/api/studio/audit/status` reports a path-free audit sink status
  for operator dashboards.
- `/api/studio/audit/export` returns a bounded, path-free JSON export of recent
  persisted audit events. The route is classified as admin-only by the Studio
  policy registry and records its own audit decision when policy enforcement is
  enabled.
- `/api/studio/jobs/status` reports path-free local worker health for operator
  dashboards. Deployments can set `SC_NEUROCORE_STUDIO_JOB_ROOT` for persistent
  per-job working directories and `SC_NEUROCORE_STUDIO_JOB_TIMEOUT_SECONDS` for
  the default cooperative timeout. Set
  `SC_NEUROCORE_STUDIO_JOB_MAX_ARTIFACT_BYTES` to cap each worker artifact.
  External EDA child processes also receive host-supported CPU and memory
  ceilings from `SC_NEUROCORE_STUDIO_EDA_PROCESS_CPU_SECONDS` and
  `SC_NEUROCORE_STUDIO_EDA_PROCESS_MEMORY_BYTES`. The local worker manager
  tracks allowed job kinds, active/completed/failed/timed-out counts, and one
  resource profile per allowed job kind. Each profile records default timeout,
  per-artifact size ceiling, and supported execution models without exposing
  host filesystem paths.
- `/api/training/start`, `/api/training/stop`, and
  `/api/training/status/{job_id}` now use the local worker manager for bounded
  training execution while preserving the training monitor's existing SSE
  metric stream contract. Terminal training jobs write `training/status.json`
  and `training/evidence.json`; the evidence manifest uses
  `studio.action-evidence.v1` and records terminal status, action kind, replay
  route, job ID, evidence classification, status payload SHA-256, and status
  artifact metadata without local paths or secret material.
- `/api/compile`, `/api/synth/run`, `/api/synth/multi-target`,
  `/api/synth/pnr`, and `/api/pipeline/run` execute through the same bounded
  local worker manager while preserving their synchronous response payloads.
  The Admin queue records `studio-compiler`, `studio-synthesis`, `studio-pnr`,
  and `studio-pipeline` owners with path-free result artifacts under
  `compiler/`, `synthesis/`, and `pipeline/`.
- `/api/compile` responses include `studio.compile-traceability.v1` metadata
  with the source equation payload, RTL module metadata, source and RTL
  SHA-256 digests, and `evidence_classification: "compile"` without
  host-local paths.
- `/api/compare`, `/api/fi-curve`, `/api/bifurcation`, `/api/sensitivity`,
  `/api/heatmap`, `/api/freq-response`, `/api/precision`, and
  `/api/nullclines` responses include `studio.analysis-result.v1` metadata
  with the analysis type, evidence classification, source, input and result
  SHA-256 digests, and output keys without host-local paths.
- Those worker-backed compile, synthesis, PnR, and pipeline routes also write
  `studio.action-evidence.v1` manifests beside the result artifacts. These
  manifests record action kind, replay route, job ID, evidence classification,
  result payload SHA-256, and result artifact metadata without local paths or
  secret material.
- `/api/studio/jobs` and `/api/studio/jobs/{job_id}` return admin-only,
  path-free job records for the Admin panel queue view. Records include job
  status, owner, request ID, timestamps, result metadata, and artifact
  manifests, but never host filesystem paths.
- `/api/studio/jobs/{job_id}/artifacts/{artifact_path}` downloads declared job
  artifacts for administrators. The server resolves artifacts through the job
  manifest only, revalidates size and SHA-256 before serving, and returns
  generic errors if the artifact is missing or fails integrity checks.
- `/api/studio/evidence/bundle` creates an admin-only evidence export as a
  bounded `studio-evidence` worker job. The request can name one saved project,
  selected job IDs, bounded audit export length, and command replay metadata.
  The resulting `studio.evidence-bundle.v1` manifest, project payload, job
  records, copied verified job artifacts, audit excerpt, and replay metadata are
  all written as job artifacts under `evidence/`. The response and manifest are
  path-free and omit bearer tokens, token hashes, password material, and local
  filesystem paths.
- The Admin panel surfaces that evidence route as an operator workflow with
  project, job ID, audit, and replay metadata fields. After export it refreshes
  the job queue and displays the bundle ID, evidence job ID, artifact count,
  and manifest entry count.
- New backend workflows can use `StudioJobManager.submit_process_task(...)` for
  process-backed execution. The task is an importable `module:function` that
  receives a `StudioJobContext` plus a JSON payload and returns a JSON result.
  It shares the same artifact manifest contract as thread-backed jobs, while
  timeout and cancellation terminate the worker process. Existing Studio route
  closures remain compatible with the thread-backed path during migration.
- `/api/studio/operator/status` is an admin-classified aggregate for the
  Studio control plane. It combines deployment profile, route-policy
  enforcement, identity mode, audit health, job-worker health, job resource
  profiles, and capability counts into one path-free payload for the Admin
  panel. It also reports path-free worker and EDA process resource ceilings so
  operators can verify runtime bounds without reading environment files.
- Deployments can set `SC_NEUROCORE_STUDIO_AUDIT_ROTATION_BYTES` to rotate the
  active JSONL audit file before the next append once it reaches the configured
  byte limit. `SC_NEUROCORE_STUDIO_AUDIT_RETAINED_FILES` controls how many
  rotated files are retained; the default is `5`.
- Audit status performs path-free storage preflight checks and reports stable
  operator error codes such as `AuditPathIsDirectory` and
  `AuditParentIsNotDirectory` instead of exposing local filesystem paths.
- `RoutePolicyRegistry` records the expected policy for each platform API route
  and lets startup or test code detect unclassified Studio endpoints before
  they become accidental public surfaces. The default registry classifies the
  current `/api/*` and `/ws/*` Studio surface, including stateful simulation,
  training, project, synthesis, and WebSocket progress routes.
- `StudioRuntimeSettings` owns deployment-sensitive backend settings. CORS
  defaults are loopback-only for the packaged backend and Vite development
  server; production deployments must set `SC_NEUROCORE_STUDIO_CORS_ORIGINS`
  to a comma-separated allow-list instead of using wildcard origins.
- `SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE=production` is a fail-closed profile:
  route-policy enforcement must be enabled, development header principals must
  be disabled, and identity, audit log, and job-root paths must be configured
  before the backend can start.
- `sc-neurocore studio-deployment-profile --studio-profile local|lab|server`
  emits a `studio.deployment-profile.v1` package for the supported operating
  contexts. `local` is loopback-only development use. `lab` and `server` are
  production-profile packages with route-policy enforcement, disabled header
  principals, durable identity/audit/job-root placeholders, explicit host and
  origin allow-lists, preflight command, launch command, saved project
  workspace, and backup items. The command can emit JSON or shell `export`
  lines and does not include secrets, token hashes, password material, or
  host-local paths.
- `sc-neurocore studio-backup-plan` emits a `studio.backup-plan.v1` backup and
  restore manifest for identity, audit, job, and saved-project state. Default
  output is path-free and suitable for deployment logs; use
  `--include-local-paths` only for internal host-local handoffs.
- `sc-neurocore studio-preflight` runs the Studio release-readiness gate from
  the current environment and emits a `studio.preflight.v1` JSON report. The
  report exits non-zero on any failed check and verifies runtime settings,
  route-policy enforcement, disabled header-principal fallback, required admin
  route policies, at least one active unexpired `studio.admin` identity,
  audit-log readiness, and job-root readiness. The required admin route set
  covers identity list/detail/update, browser-user create/update, and password
  rotation routes, job record and artifact access, and evidence bundle export
  in addition to operator/audit/synthesis checks. Its payload contains stable
  check IDs, booleans, counts, and path-free remediation steps; it does not
  expose local paths, bearer tokens, token hashes, passwords, or password
  verifiers.
- Studio rejects requests whose `Host` header is outside the configured
  allow-list. Packaged defaults accept only loopback hosts; deployments set
  `SC_NEUROCORE_STUDIO_ALLOWED_HOSTS` to a comma-separated host allow-list and
  wildcard hosts are rejected.
- Studio rejects HTTP requests with a declared body larger than the configured
  limit before route handlers run. The default is 1 MiB; deployments can set
  `SC_NEUROCORE_STUDIO_MAX_REQUEST_BODY_BYTES` to a positive integer.
- Studio WebSocket handshakes enforce an explicit `Origin` allow-list before
  accepting progress streams. By default this allow-list follows the HTTP CORS
  origins; deployments can set
  `SC_NEUROCORE_STUDIO_WEBSOCKET_ALLOWED_ORIGINS` to a comma-separated list.
- The backend adds default HTTP hardening headers to API responses:
  `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, and
  `X-Frame-Options: DENY`.
- Every HTTP response carries an `X-Request-ID` correlation header. Valid
  inbound request IDs are preserved; malformed values are replaced with a
  server-generated UUID.

The policy gateway is intentionally a platform contract first. Existing public
health and capability routes stay backward compatible while protected route
integration is added panel by panel.

## Development Gate

Studio changes are shipped as complete slices. Each backend slice updates
strict typing, public-symbol docstrings, module-specific tests, focused coverage
for new modules, and documentation in the same change. Generic coverage-fill
tests are not accepted.

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
generated Verilog side-by-side with a verification badge. Direct
equation-to-Verilog output includes a source-to-RTL traceability strip with
source, RTL, and manifest digests for evidence review.

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
Synthesis responses include path-free target provenance with tool readiness,
tool versions where available, capacity metadata, and a multi-target matrix
digest for comparison runs.

### Full Pipeline

One-click pipeline from the Network Canvas:

```
Network Graph → Validate → Simulate → Compile → Synthesise
```

### Project Save/Load

Save complete workspace state (equations, parameters, network graph,
synthesis target, training config) as JSON files. Restore any saved
project from the sidebar. The save API returns `studio.project-save.v1`
metadata with state and project SHA-256 digests plus the
`project_workspace` evidence classification; it does not expose resolved
local filesystem paths.

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
| Frontend tests | Vitest contract tests + TypeScript build |
| Backend | FastAPI |
| Simulation | SC-NeuroCore Python + Rust engine |
| Training | PyTorch (optional `[research]` extra) |
| Synthesis | Yosys + nextpnr (external, optional) |

## API Reference

The Studio exposes 40+ REST endpoints. See the per-block documentation
for complete API details with request/response examples.

| Prefix | Block | Endpoints |
|--------|-------|-----------|
| `/api/simulate`, `/api/models/*` | Phase 1 | ODE/model simulation with path-free run metadata, templates, presets |
| `/api/fi-curve`, `/api/bifurcation`, `/api/heatmap`, ... | Phase 1 | Analysis views with path-free result metadata |
| `/api/ir/*` | Compiler Inspector | IR build, verify, emit SV, co-sim, direct-SV traceability |
| `/api/compile`, `/api/synth/*` | Compiler/Synthesis | Worker-backed compile traceability, Yosys synthesis, target provenance, multi-target, estimate |
| `/api/training/*` | Training Monitor | Start/stop, SSE stream, surrogates |
| `/api/graph/*` | Network Canvas | Populations, projections, validate, simulate, NIR |
| `/api/project/*`, `/api/pipeline/*` | Integration | Save/load, worker-backed full pipeline |
| `/api/studio/audit/*` | Admin | Audit status and admin-gated export |
| `/api/studio/evidence/bundle` | Admin | Job-backed project/job/audit evidence bundle |
| `/api/studio/auth/*` | Platform | Browser-user login, current bearer session, logout |
| `/api/studio/identity/*` | Admin | Token-free service-account inventory and role updates |
| `/api/studio/operator/status` | Admin | Aggregate operator control-plane health |

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
