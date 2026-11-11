<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Standalone Linux desktop wheel assembly -->

# Assemble the Linux desktop runtime

The wheel assembly command replaces the complete distribution set in an existing
standalone Python build directory. It uses the original wheel archives and PyPA
installer to generate relative console launchers and their installed `RECORD`
entries. Run it before creating the desktop bundle manifest or delivery archive.

Use a separate build interpreter with the project's locked build dependencies.
The target must contain its own Python executable and standard library. A virtual
environment whose base Python is outside the target is refused. The control
interpreter, uv executable, wheel inputs and receipts must all be outside the
target. Managed installed desktop versions are refused; use build staging.

The producer supports Linux launchers. The execution host needs `/usr/bin/env`
with `-S` support, `/bin/sh`, `dirname` and `realpath`. A single comment-only
shebang dispatches to the adjacent interpreter; the original Python body stays
unchanged, including encoding declarations, module docstrings and future imports.
See the [GNU env shebang contract](https://www.gnu.org/s/coreutils/manual/html_node/env-invocation.html).
All resource and whole-job limits belong to the admitted
build host profile; the command also requires a finite deadline for each child.

## Select the exact archives

Acquire wheels using the selected hashlocked runtime and federation requirements,
the exact core/native/SDK build inputs and the selected seed tooling. Include
every distribution already installed in the target, including packaging tools.
There must be exactly one wheel per normalised distribution name. Source builds
and channel substitution are separate decisions; this command neither resolves
dependencies nor rebuilds wheels.

Place the selected archives beside a UTF-8 JSON catalog. Its fields are:

| Field | Required value |
| --- | --- |
| `schema_version` | `studio.desktop.wheel-catalog.v1` |
| `python_version` | The target's full stable Python version, at least 3.11 |
| `platform` | The target's actual Linux `sysconfig.get_platform()` value |
| `wheels` | Nonempty array with one exact archive declaration per distribution |

Each wheel declaration has precisely `filename`, `sha256` and `size_bytes`.
`filename` is one archive basename, `sha256` is its complete lowercase SHA-256 and
`size_bytes` is its positive integer byte length. Symbolic links, duplicate JSON
fields, unknown fields and duplicate distributions are refused. The reader also
checks the wheel's original member hashes, distribution metadata and declared
Python requirement. The isolated target inventory probe uses only Python's
standard library. Wheel compatibility comes from the selected runtime's actual
`python -I -B -m pip --isolated debug --verbose` command, with its original output retained.
The child uses `PIP_CONFIG_FILE` set to the operating system's null device;
ambient configuration and pip option variables cannot select foreign tags.
The parser checks the complete reported tag count and refuses a changed output
contract. The selected pip archive is part of the same pinned input closure;
the target does not need the build interpreter's `packaging` dependency.
Pip documents this diagnostic output as
[provisional](https://pip.pypa.io/en/stable/cli/pip_debug/); requalify this
consumer when changing the selected pip archive.

`inspect_runtime()` executes both diagnostics and validates the live installed
files. Its public `parse_runtime_inventory()` decoder can also validate captured
report bytes against a selected catalog. Decoding a report alone does not prove
that the current runtime still contains those files; retain the live probe and
its command receipts for assembly qualification.

After normal hashlocked acquisition, create the catalog with:

```bash
python -m tools.studio_desktop_wheel_catalog \
  --directory "$selected_wheels" \
  --output "$selected_wheels/catalog.json" \
  --python-version "$actual_python_version" \
  --platform "$actual_python_platform"
```

The command includes every `.whl` archive in that directory and validates it
before creating the new file. An existing output is refused. The interpreter
values must come from the actual target. This snapshots already selected inputs;
dependency resolution, trusted source selection and lock verification remain
the acquisition producer's responsibility.

The exact upstream PySide6 Essentials 6.12.0 Linux wheel includes two generated
bytecode cache members that PyPA installer 1.1.0 does not install. The shipped
`tools/studio_desktop_generated_bytecode.toml` binds only that complete archive
SHA-256, those two paths, their original RECORD hashes and sizes, and their
RECORD-verified Python source companions. Every original archive member remains
integrity-checked. Any other cache member or archive pin is refused. A pin change
requires requalification; a cache-free upstream wheel needs no exception.

## Inspect, then apply

Run the tools module from the repository root, using the separate build
interpreter. Substitute the paths and timeout from the admitted build job:

```bash
python -m tools.studio_desktop_install_wheels \
  --runtime "$runtime_prefix" \
  --catalog "$wheel_catalog" \
  --uv "$uv_executable" \
  --receipts "$plan_receipts" \
  --timeout-seconds "$command_timeout"
```

Without `--apply`, the command verifies the complete installed inventory and
every installed `RECORD`, preserves the original records and runs uv's normal
uninstall dry run. It writes `plan.json` and leaves the target unchanged. Receipt
directories must be new; an existing directory is never overwritten.

To perform replacement, repeat with `--apply` and a different new receipt
directory. This performs a normal named uninstall before writing new files,
then installs the complete original wheel set through PyPA installer. Unlinking
old files first avoids truncating an existing hard link into a shared wheel
cache. Installed metadata records the standard installer identity and contains
no local source URL.

Before uninstalling, the producer also plans shared file ownership from the
complete original closure. Multiple library owners must declare identical
payload hashes, sizes and executable modes. Shared generated scripts are
refused. Each admitted file is created once through the normal installer; a
later owner may retain it only if this same destination instance created it
and its inode, link count, mode and bytes still match. The incoming bytes must
also match. Each owner receives the normal `RECORD` entry without rewriting
the shared file. An unrelated existing file, another destination instance,
symbolic link or mismatch is refused. The broad overwrite option is refused.
The plan and result record every admitted shared path and its archive owners.
Each owner retains ordinary `RECORD` uninstall semantics: uninstalling one
owner removes its shared path. Updates must therefore replace the complete
selected distribution set.

After installation, the command checks the exact package set, all installed
records, original wheel payloads, transformed Python data-script bodies,
launcher prefixes and `uv pip check`. It rechecks all original archive bytes.
Only a complete successful replacement writes `result.json` with
`"state": "installed"` under `studio.desktop.runtime-assembly.v2`. The result
separately records all original member bindings, the exact non-installed
upstream bytecode dispositions and the policy file SHA-256. Normal PyPA output
must not contain those two caches; another installation path retaining them
must match their original bytes. No original wheel or RECORD is edited.
Original command arguments, stdout, stderr and process
cleanup observations are retained for each child command.

Replacement is a staging operation rather than an atomic installed-app update.
A failed apply can leave the staging runtime incomplete. Retain its diagnostics
and repair or recreate the owned staging input before trying again. Use the
desktop installer's separate update and rollback operations for installed user
versions.

## Qualify the resulting runtime

An assembly result always reports `bundle_sealed: false` and
`desktop_acceptance: false`. Move the complete staging runtime to another owned
path with the original prefix unavailable, then exercise its real generated
entry points and core/native/SDK/Qt imports. Verify the installed records again
after movement. Successful archive integrity alone does not prove relocation.

After actual runtime qualification, create and verify the enclosing bundle with
`sc_neurocore.studio.desktop.bundle`, then create its delivery archive with
`sc_neurocore.studio.desktop.archive`. Final acceptance uses those exact bytes:
application launch, compilation, save/restart/reopen, export, update and
data-preserving rollback. SDK and federation-host acceptance remain separate
requirements. Runtime assembly does not change the Studio interface or its
layout.

## Test inputs

The complete runtime cohort lives in `studio/desktop/assembly_tests`, with one
dedicated test module for each assembly tool. It runs through the reusable
`.github/workflows/studio-desktop-runtime.yml` lane. The normal test-tree
discovery script remains unchanged.

One producer acquires the original standalone interpreter and the exact runtime,
script-format and foreign-ABI archive profiles. It creates the three catalogs
and shares one input artifact with a checksum manifest and source revision.
The consumer requires that producer to succeed, downloads its exact artifact ID,
and verifies the manifest digest, complete file set, every file checksum and
matching source revision before using any input. An artifact digest mismatch
fails the download.

The consumer uses `requirements/studio-assembly-tools-linux-x86_64.txt` for its
complete controller environment. Its adjacent `.in` file selects the maintained
profiles and pinned acquisition tools; the generated `.txt` retains exact
versions and hashes for the blocking dependency audit. The runtime and two
archive-control profiles are separate locks. Existing first-party baseline
wheels are test inputs, separate from artifacts built for a new release.

The consumer seeds one standalone runtime offline through normal hashlocked uv
installation, including a same-version reinstall of bootstrap pip. It compares
actual collection with executed JUnit cases and requires zero skips, failures
or errors. `studio/desktop/assembly-coverage.ini` covers all six complete tool
modules, including branches and subprocesses, with no exclusions and a 100%
threshold. The input job has a 15-minute ceiling and the consumer a 60-minute
ceiling; bounded child commands use the remaining job time after setup and a
two-minute diagnostics reserve. The lane retains original commands, output,
process receipts, membership and coverage data.

The dedicated wheel-catalog tests require
`SC_NEUROCORE_STUDIO_ASSEMBLY_INPUTS` to name the directory containing the exact
published pip 26.2.2 and locked Pint 0.26.2 wheel archives. Their full SHA-256
values are checked before each test uses a private copy. The Pint wheel's
original Python requirement provides the real incompatible-version control;
no substitute wheel is generated. Missing inputs fail the tests. Acquisition
must happen explicitly in the owning CI producer before the offline test cohort.
Negative metadata tests copy the complete original pip archive and alter one
declared field, updating only that member's archive `RECORD` binding. All other
members must remain byte-identical. These deliberately malformed copies test
layout and duplicate metadata refusals; they are never runtime installation
inputs. The selected published originals remain unchanged.

The whole-runtime tests additionally require
`SC_NEUROCORE_STUDIO_ASSEMBLY_RUNTIME` (the owned standalone staging prefix),
`SC_NEUROCORE_STUDIO_ASSEMBLY_CATALOG` (its complete adjacent-archive catalog) and
`SC_NEUROCORE_STUDIO_ASSEMBLY_COMMAND_SECONDS` (the admitted job's finite positive
command deadline). The locked control environment must be a separate complete
virtual environment and provide uv. These fixtures are loaded only by tests that
request them. A runtime fault test renames the original file before writing a
replacement and restores the original inode on exit, preserving shared cache
hard links. Never point this cohort at an installed user version.

The data-script tests also require
`SC_NEUROCORE_STUDIO_ASSEMBLY_SCRIPT_CATALOG`, binding these genuine archive
format controls to the same Python version and platform:

| Archive | SHA-256 |
| --- | --- |
| `docutils-0.20.1-py3-none-any.whl` | `96f387a2c5562db4476f09f13bbab2192e764cac08ebbf3a34a95d9b1e4a59d6` |
| `ninja-1.13.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl` | `65a24341b5ac09fcadcc37082660be40a94174e51a937fabf6e2cae26225fa2c` |

Docutils supplies original generic Python data scripts and Ninja supplies an
original ELF data script. These are test inputs, separate from desktop runtime
dependencies. A function-scoped fixture installs both through the production
destination into owned staging, checks the complete augmented inventory and
dependency set, and exercises the real commands. Teardown normally uninstalls
both distributions and checks the original complete package set, unchanged
`RECORD` hashes and original wheel payloads. Missing inputs fail the test.

The command-custody tests use the repository's existing kernel-refusal support
to exercise an actual refused process-group signal. A separate process-handle
signal then terminates and collects that worker. A copied qualification tree
may select the original support file with
`SC_NEUROCORE_STUDIO_ASSEMBLY_REFUSAL_SUPPORT`; the normal repository layout
uses `tests/studio_seccomp_support.py`.

When coverage is active, the probe tests load the control interpreter's coverage
package explicitly inside the actual target interpreter. Both interpreters must
use the same Python ABI. The target's import paths stay unchanged, no coverage
distribution is installed there, and coverage data stays outside the runtime.
The traced probe output and installed-record hashes are compared with a direct
isolated invocation. Both the command entry point and the public `main()` call
read the actual complete distribution set. Kernel-refusal and runtime-fault
tests retain original output and restore the owned staging input on exit.

Installer file-drift controls run in separate controller processes. A Python
subprocess audit event marks the interval after the real RECORD probe has exited
and before the actual pip tag diagnostic starts. The controller changes one real
file there; every subprocess and producer function still executes normally.
The pre-install control changes an actual installed RECORD and confirms refusal
before uninstall. The post-install controls reuse origin metadata and an
absolute launcher produced by a normal offline hashlocked uv installation of
the original pip wheel. The complete producer then reinstalls its closure,
and the controller reintroduces those captured bytes after the final probe.
Each case requires the specific refusal, restores the temporary change, and
rechecks the complete original package set, RECORD hashes and wheel payloads.
The audit hook exists only in that controller process.
