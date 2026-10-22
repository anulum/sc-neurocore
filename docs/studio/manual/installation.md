<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — SNN Studio user manual -->

# Install and launch

## Choose how you will use Studio

| Delivery | What you open | Requirements and boundary |
|---|---|---|
| Installed Python package | A page served by your local Studio process | Python environment and the `studio` extra; a wheel with built UI assets |
| Source checkout | The built UI, or a frontend development server | Developer toolchain and a UI build; see the [technical guide](../../guides/studio.md#development) |
| Federated host | The Studio panel supplied by your host | The host's admitted plugin/backend contract and operator configuration |
| Linux desktop preview | A native window with a bundled backend/runtime | Under qualification; this manual does not identify a released desktop installer or promise clean-machine compatibility |

The packaged browser interface does not need Node.js to run. Node.js is needed
when building the frontend from source. Optional training, compiler and FPGA
features have additional requirements; install them for a task that needs them,
rather than treating every disabled panel as a failed basic installation.

## Install the browser-based application

Use a supported Python version from the
[installation profiles](../../guides/install_profiles.md). In an environment
managed with `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install 'sc-neurocore[studio]'
sc-neurocore studio
```

Keep that environment activated when using the commands in this manual.
Record the installed version when reproducing another person's experiment:

```bash
sc-neurocore --version
```

The browser opens after the server starts listening:
`http://127.0.0.1:8001/studios/sc-neurocore/`.
The root address redirects to the interface. Keep the terminal open while using
Studio. To stop the browser-based server, return to that terminal and press
**Ctrl+C**; closing its browser tab alone does not stop the server.

If port 8001 is occupied, inspect the existing process or choose another port:

```bash
sc-neurocore studio --port 9000
```

A different port is a different browser-storage origin. Your server projects can
still be in the same directory, while the **Sessions** list may differ.
Do not change ports merely to look for a missing browser session.

If the UI assets are absent, development launch falls back to API documentation.
Lab/production profiles expose `/openapi.json` instead of interactive API docs.
Seeing API documentation is not the same as loading the Studio interface.
Use a wheel containing the built frontend or follow the source build procedure.

## First launch

1. Complete or dismiss the onboarding tour.
2. Wait for the model and experiment libraries to load. Check the backend
   connection if they are missing, and use **Retry** where the interface offers it.
3. Read the workbench and readiness rows in the sidebar.
4. Select **Models** and follow the [first experiment](first-experiment.md).

A previously displayed entry can be stale after a failed refresh. Check that
the requested model and its details loaded successfully before running it.
Preserve work before reloading the page; a different model is not a substitute
for a failed request.

## Where your projects are stored

The browser launcher uses `~/.sc-neurocore/studio/projects` by default.
To choose a directory, set an absolute path before launch:

```bash
export SC_NEUROCORE_STUDIO_PROJECT_ROOT="$HOME/Research/studio-projects"
sc-neurocore studio
```

The shell expands `$HOME` here. A literal value such as `~/Research/projects`
is refused. Choosing a directory does not move existing projects.
Stop Studio before a deliberate move or restore, then verify the project list
under the new configuration. Include hidden recovery data in backups; see
[Saving and recovery](projects-and-evidence.md#back-up-and-restore).

The Linux desktop preview separates installation resources from user data.
Its default configuration, data, state and cache bases follow the Linux XDG
locations, each under `sc-neurocore/studio`. It can retain an existing legacy
project directory; it does not silently move it. A desktop browser profile also
has its own sessions. Browser and desktop storage are not interchangeable
merely because both display the same project name.

## Operator and lab use

The default unprotected local-development launch is separate from the supported
single-user loopback lab configuration. For identity-based use, the operator
must configure the identity store, route policies, allowed origins, audit log
and one API process. Follow the complete
[lab setup](../admin-identity.md); setting a deployment label alone is insufficient.

The operator commands expose deployment and recovery information:

```bash
sc-neurocore studio-deployment-profile --studio-profile lab
sc-neurocore studio-preflight
sc-neurocore studio-backup-plan
```

A generated profile describes configuration. A preflight report checks the
configured environment. Neither is a security approval or a claim that every
research, compiler and hardware workflow has passed. Admin routes can be
unavailable until an identity store is configured even when simulation works.
Keep login credentials and one-time service tokens out of shared reports.

Shared multi-process sessions, OIDC and exposing Studio on a LAN/VPN require a
separately reviewed operating contract. Do not expose the development process
by changing the bind address or relaxing origin checks.

## Federated Studio

A host-embedded panel also depends on the host's version, capabilities and
backend bridge. A working standalone page does not establish host adoption.
Use the [federation reference](../../api/federation.md) for the exact integration
contract. If the host panel is unavailable but standalone Studio works, report
the host and plugin versions to the host operator; do not forward credentials
or alter sandbox/origin policy to make the panel load.

Next: [Run your first experiment](first-experiment.md).
