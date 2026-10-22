<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — SNN Studio user manual -->

# Troubleshooting

Start by preserving the current project or export if the page still works.
Read the failing panel's own message and note the action, selected model/source,
installed version and job ID. Avoid clearing browser data or deleting user
files as an initial remedy.

## Launch, libraries and login

| Symptom | What to check | Safe next step |
|---|---|---|
| `sc-neurocore` is not found | Activated environment and installed package | Activate the environment used for installation |
| Missing FastAPI/Uvicorn | `studio` extra in that environment | Install the extra with the same environment's package manager |
| Port is in use | Another Studio or service owns it | Stop only a process you own, or choose `--port 9000`; remember the session origin changes |
| API documentation opens instead of the UI | Built frontend assets are absent | Use a wheel containing the UI or build the frontend from source |
| Library missing or request fails | Backend connection and the failed request | Use **Retry** where offered; preserve work before reloading, and check that model details match the selection |
| ODE editor fails while other panels load | Local editor assets, build and content policy | Report the asset error/build to the operator; do not disable policy or fetch arbitrary scripts |
| Admin is unavailable but simulation works | Identity store and enforced policy | Ask the operator to follow the complete lab identity setup |
| Login/authorisation refused | Correct deployment, account, expiry and role | Reauthenticate through the admitted login; do not place a bearer token in a URL |
| Standalone works, host panel fails | Host/plugin/backend versions and capabilities | Give the host operator those versions; preserve sandbox and origin checks |

## Simulation and analysis

| Symptom | Likely checks, not a diagnosis | Next step |
|---|---|---|
| No spikes | Drive, units, initial state, threshold, duration and selected model | Compare with a known small experiment; keep a silent successful run distinct from a failed request |
| State becomes non-finite | Timestep, singular equations and parameter domain | Save the request; correct its numerical configuration deliberately |
| Seed refused | Deterministic source or model-specific generator limit | Remove an inapplicable seed or use the documented range |
| Large run refused | Step/neuron/input-memory budgets | Reduce work or use the offered job path; do not assume the server shortened it |
| Old plot remains after failure | Plot identity versus current run status | Read the failure; rerun successfully before marking the experiment complete |
| Phase/ISI tab absent | Required state variables or spike intervals | Obtain an applicable successful result; the tab is context-dependent |
| Metric is undefined | Too few spikes, incompatible inputs or invalid analysis contract | Preserve the reason and adjust the experiment, not the reported value |
| Raw export is missing or bounded | Element budget and omission reason | Reduce work for full custody, or retain the explicitly limited result |
| Async result never becomes current | Changed experiment, cancellation or connection loss | Inspect its retained job status and rerun for current inputs if necessary |

## Projects, downloads and replay

| Symptom | Check | Next step |
|---|---|---|
| Project missing | Project-root configuration, list refresh, deleted list | Check **Refresh** and **Deleted (restorable)** before creating another copy |
| Sessions missing | Browser profile, host, scheme, port or desktop profile | Return to the original origin; server projects are separate |
| Session storage warning | Browser persistence failed | Save a server project/export while the page retains the edit |
| Save conflict / HTTP 409 | Another revision became current | Preserve the draft; compare/fork through workspace recovery |
| `workspace_busy` / HTTP 503 | Another writer holds the workspace | Retry after it finishes; do not delete lock files |
| Restore refused | The original name is already in use | Preserve both live and trashed workspaces and resolve the name deliberately |
| Download says success but no file found | Browser save destination, permission or cancellation | Check Downloads and retry; the UI cannot certify disk persistence |
| Artifact hash/size mismatch | Missing, changed or truncated bytes | Retain the error and retry; do not bypass integrity checks |
| Replay exit 1 | Recorded differences | Compare inputs, state and environment; report the mismatch |
| Replay exit 2 | Admission refusal | Read identity/runtime/schema reason before changing options |

## Training, compiler and hardware

A disconnected training stream is not proof that the worker stopped. Use
**Refresh runs** and inspect the durable status. `unknown` means uncertain
liveness; `interrupted` records a lost supervisor; neither means completed.
Stop is cooperative. A failed input-manifest check requires correcting the
recordings/contract, not silently substituting another dataset or decoder.

For compilation, distinguish catalogue-schema RTL, direct Python equation
compilation and the Rust IR inspector. The current fixed-template Rust adapter
cannot establish translation fidelity for arbitrary equations. See the
[hardware limitation](hardware.md#custom-equations-and-the-compiler-inspector).
Do not treat emitted text or a structural verifier badge as a successful
simulation, co-simulation or synthesis.

For synthesis failures, preserve the exact source, target, tool versions,
parity prerequisite and job outcome. Check missing Yosys/nextpnr, unsupported
target, timeout, capacity, uncounted cells and I/O constraints. A heuristic
estimate does not repair a failed real tool run.

For a desktop preview refusal, keep the original configuration and retained
data contract. An unsupported/future schema or different data family can
correctly prevent startup, update or rollback. Close the owned application
before lifecycle operations. Do not delete a family stamp, force a selector
or remove user projects to get past the refusal; retain a backup and ask the
operator to inspect the compatible build/schema boundary.

## Report a reproducible issue

Include:

1. Installed version, delivery (browser/source/desktop/host), OS and browser.
2. Model name or equations, numerical inputs, protocol, seed/trial and graph
   or training configuration relevant to the failure.
3. The exact action and authored message, job ID and current status.
4. The smallest project/replay pack/evidence bundle that reproduces it.
5. Expected versus observed result, with any raw-state or metric limitations.

Remove credentials, service tokens, private host paths and restricted recordings
from material you intend to share. Keep the original local evidence intact.
Do not report a successful paper benchmark, production approval or physical
hardware result merely because a development workflow completes.

Next: [Glossary](glossary.md), or return to the [manual contents](index.md).
