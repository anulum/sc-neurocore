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
| Linux desktop from source | A native window with a local backend | Python, the `studio` and `desktop` extras, system graphics libraries and built UI assets |
| Bundled Linux desktop preview | A native window with a bundled backend/runtime | An operator-supplied complete Linux x86_64 bundle; see [bundled installation](#install-a-bundled-linux-desktop-preview). A public desktop release is still pending |

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

## Launch a native Linux window from source

The source desktop uses the same Studio interface inside its own Qt window.
The first desktop target is Linux x86_64. The qualified dependency profile uses
CPython 3.12 and Qt 6.12; it does not establish support for every Linux distribution.
A graphical login and the system libraries required by
[Qt on Linux](https://doc.qt.io/qt-6/linux-requirements.html) are necessary.
A missing system library is separate from a missing Python package.

From the repository root, use a dedicated Python environment. Build the UI using
the repository's locked frontend dependencies and qualified Node.js 26.9.0
profile, then install the desktop extras:

```bash
uv venv --python 3.12 .venv-studio
source .venv-studio/bin/activate
(cd studio/frontend && npm ci && npm run build)
uv pip install --require-hashes --only-binary :all: \
  -r requirements/studio-desktop-linux-x86_64.txt
uv pip install -e '.[studio,desktop]'
snn-studio
```

The equivalent module command is:

```bash
python -m sc_neurocore.studio.desktop
```

The window first displays **Starting Studio…**, then opens the interface after
its backend and UI assets have been checked. The desktop retains its local port
in the profile, so reopening the same profile retains the browser-storage origin.
If another application occupies that port, close that application or resolve the
conflict before retrying. Two instances cannot own the same desktop profile.

Use **File → Quit** or the window's close button to stop the desktop and its
owned backend. Keep the window open while **Closing Studio…** is displayed.
If cleanup cannot be confirmed, Studio keeps a visible retry message; use
**Close** again. Normal shutdown releases the profile and retains your projects
and configuration. Closing a separately opened browser tab does not send this
desktop shutdown request.

A missing-Qt message means the selected interpreter lacks the desktop runtime;
activate the environment where you installed it. A directory-permissions refusal
means the selected XDG/profile directories cannot be prepared. Do not delete an
existing profile to clear either error. Startup, page-load and renderer failures
remain visible in the native window. When available, its diagnostic log is
`$XDG_STATE_HOME/sc-neurocore/studio/desktop.log`, or
`~/.local/state/sc-neurocore/studio/desktop.log` when that variable is unset.

This source workflow requires an existing Python environment. The bundled
workflow below carries its own interpreter and registers a menu entry.
Optional scientific compiler and physical-hardware capabilities retain their
own requirements and readiness indicators.

## Install a bundled Linux desktop preview

These steps apply to a complete preview bundle supplied by your operator.
A public desktop installer is not available yet. The first qualification target
is Linux x86_64; testing on Ubuntu 24.04 does not establish compatibility with
every distribution. The bundle includes Python, the built interface and its
Python/Qt dependencies. It still needs the target system's graphics libraries
and a graphical login. You do not need a source checkout, Node.js or a separately
installed developer Python to use this delivery.

Extract the complete supplied archive into an ordinary directory outside your
Studio application or project directories. Keep its `runtime`, `share`,
`bundle.json` and `install.sh` together. From that extracted directory, run:

```bash
./install.sh
```

Run this as your normal desktop user. The installer copies the complete bundle
into your user application directory, verifies it, registers **SNN Studio** in
the application menu and selects the installed version. It prints a version and
a full build identity; keep that output with your experiment notes. Installation
does not open the application automatically.

Open **SNN Studio** from the desktop's application menu, or use:

```bash
"$HOME/.local/bin/snn-studio"
```

The menu entry and command open the native window described above. If your shell
already includes `~/.local/bin` in `PATH`, you can also use `snn-studio` directly.
The installed application runs from its own copy; the extracted preview directory
does not need to remain open. Keep the supplied archive separately if you need
to reproduce or repair that exact build.

### Update and return to a retained version

Save your work and quit Studio before changing the application. Extract the new
complete bundle into a separate directory. Run its `./install.sh`, or give its
absolute directory to the installed manager:

```bash
"$HOME/.local/bin/snn-studio-manage" install "/absolute/path/to/new-bundle"
```

The manager verifies the new copy before changing the selected version. It
retains earlier installed versions. Reinstalling the identical bundle is safe
and does not create another version. Keep the same HOME/XDG configuration for
installation, launch and management commands so they address the same profile.

To inspect retained identities and the selected version:

```bash
application="${XDG_DATA_HOME:-$HOME/.local/share}/sc-neurocore/studio/application"
readlink "$application/current"
ls "$application/versions"
```

`current` names `versions/` followed by the selected build's full identity.
The version directory names are content identities, not dates or version numbers.
Use the identity printed by the earlier installation you want to select. Replace
the placeholder below with that complete 64-character identity:

```bash
"$HOME/.local/bin/snn-studio-manage" rollback FULL_64_CHARACTER_BUILD_ID
```

Rollback selects a verified retained application with a compatible data contract.
It does not restore an earlier project, undo an experiment or migrate newer data
into an older format. Restore research data from your own backup when that is
your intention. An incompatible data contract requires a separately qualified
migration; there is no force option to bypass this check.

### Uninstall while retaining research

Save your work and quit Studio, then run:

```bash
"$HOME/.local/bin/snn-studio-manage" uninstall
```

This removes the managed application, all its retained code versions, its two
command launchers and its menu entry. It retains projects, desktop settings,
browser sessions, evidence, state and cache. A later installation must still be
compatible with that retained profile. Uninstall is not a research-data reset.

With default XDG locations, the profile uses:

| Location | Contents |
|---|---|
| `~/.config/sc-neurocore/studio` | Desktop configuration |
| `~/.local/share/sc-neurocore/studio` | Application data, browser profile and default projects; the `application` subdirectory contains managed code |
| `~/.local/state/sc-neurocore/studio` | Runtime state, logs and retained compatibility record |
| `~/.cache/sc-neurocore/studio` | Application cache |

An existing legacy project directory or an explicitly configured project path
can be elsewhere. Include it when following [backup and restore](projects-and-evidence.md#back-up-and-restore).

### If the installer refuses a change

| Message or condition | What to do |
|---|---|
| Close Studio or finish another installation | Quit through the native window, wait for shutdown and retry when the other operation finishes |
| Bundle contents changed, missing runtime or invalid identity | Use a fresh complete copy of the exact supplied bundle; do not combine files from different builds |
| An existing launch file or symlink would be replaced | Preserve the reported file and ask the operator to resolve its ownership before retrying |
| A version needs a data migration or a newer configuration schema | Keep the profile intact and use its compatible application; arrange a qualified migration before switching families |
| Unowned files, an invalid selector or changed installed payload | Preserve the directory and research backup for inspection; do not delete ownership records to make the check pass |

The installer does not repair foreign files by overwriting them. Research data
and application code have different recovery procedures. If you need help,
retain the exact refusal, requested operation, supplied version/build identity
and target Linux distribution; keep credentials out of shared reports.

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

### Export a complete desktop bundle

Operators can export an existing sealed preview bundle from an environment
containing the desktop archive exporter:

```bash
python -m sc_neurocore.studio.desktop.archive \
  "/absolute/path/to/sealed-bundle" "/absolute/path/to/snn-studio.tar.gz"
```

Choose a new output filename outside the input bundle. The command checks every
sealed entry, retains its permissions and internal links, then prints the archive's
SHA-256, size and original bundle build identity. It refuses an existing output
file, a changed input or a failed write. A failed export leaves no partial output
at the requested filename. Extracted contents are the bundle itself, with
`install.sh` at the extraction root.

Two exports of the same sealed input with the same Python/zlib toolchain produce
identical archive bytes, independent of output filename or export time. This
does not rebuild dependencies or qualify their origin. Retain the actual runtime,
source and dependency provenance separately, and supply the archive checksum
through a trusted channel. A checksum alone does not authenticate its publisher.
Public desktop release, signing, a declared minimum Linux baseline and complete
compiler acceptance remain separate requirements.

When building the frontend from source, `npm run build` also normalizes the ZIP
entry timestamps in the generated federation declarations. It retains the type
names, declaration contents, permissions and comments; a different build time or
host timezone therefore does not change that archive. The frontend's
`npm run test:build` command checks the real archive command, preservation and
refusals with complete coverage of the normalizer. Dependency versions, the
Node.js toolchain and all other source inputs still need their own provenance.

### Configure a lab instance

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
