# Maintenance Tools

This page records repository maintenance tools that produce audit evidence but
do not change runtime behaviour. Each tool emits timestamped artefacts where
possible, so historical audits remain reproducible.

## 2026-04-30 Tooling Baseline

### Compiler HDL e2e workflow

The `Compiler HDL E2E` workflow runs the `tests/e2e/` corpus for pull requests
that touch the compiler package, HDL generator package, e2e tests, or the
workflow itself. The lane is intentionally path-filtered and PR-only so compiler
or HDL generator changes get full cross-surface integration coverage without
duplicating the whole default CI matrix.

Run the same selector locally after changing `src/sc_neurocore/compiler/`,
`src/sc_neurocore/hdl_gen/`, or `tests/e2e/`:

```bash
PYTHONPATH=src:. python -m pytest tests/e2e/ -m e2e -q
```

The workflow contract is covered by:

```bash
PYTHONPATH=src:. python -m pytest tests/test_tools/test_compiler_e2e_workflow_contract.py -q
```

### Test inventory audit

`tools/test_inventory_audit.py` compares tracked `test_*.py` files with a
pytest collect-only transcript. The monthly
[Audit Cadence](audit_cadence.md) workflow uses it to detect test-inventory
drift without running the full suite outside the normal CI matrix.

Run it locally after adding, moving, or optional-gating test files:

```bash
PYTHONPATH=src:. python -m pytest tests/ --collect-only -q | tee audit-collect-only.txt
python tools/test_inventory_audit.py \
  --repo . \
  --collect-output audit-collect-only.txt \
  --output audit-inventory.json
```

Tracked files absent from base collection must declare a module-level
`pytest.importorskip(...)`, an explicit
`pytest.skip(..., allow_module_level=True)` guarded by
`importlib.util.find_spec("dependency") is None`, or the standard `__test__ = False`
marker for a compatibility or shared-support module. The JSON records each
classification; any unexplained uncollected tracked test file is a failure.

Missing-reference guards may combine the absence check with stricter conditions
using `and`. Broader `or` guards, installed-module checks and function-only skips
do not explain an uncollected module. Required TorchHD reference jobs still fail
when the reference is missing or broken.

### SNN memory-discipline audit

`tools/snn_memory_discipline_audit.py` audits SC-NeuroCore SNN stimulus
producers and emitted JSON records against the fleet memory-write schema from
`BROADCAST_2026-06-29_memory_write_discipline.md`. It discovers tracked Python
writer functions, validates every `*.json` stimulus file in the selected
directory, and reports noncanonical keys, uncontrolled actor roles, missing
timestamps, missing entities, and empty provenance.

Run the audit against the shared SC-NeuroCore stimulus directory:

```bash
python tools/snn_memory_discipline_audit.py \
  --repo . \
  --stimulus-dir "$SC_NEUROCORE_GOTM_ROOT/04_ARCANE_SAPIENCE/snn_stimuli/SC-NEUROCORE" \
  --output docs/internal/snn_memory_discipline_audit.json
```

Use `--repair` only for schema-only normalization of legacy records. The repair
path preserves factual content by moving legacy `summary`, `commit`,
`todo_rows_closed`, and `evidence` fields into canonical `content` and
`source_ref` fields; it does not delete stimulus files.

### Systematic audit rerun

`tests/test_tools/test_systematic_audit_rerun_contract.py` keeps the concrete
findings from `docs/internal/audit_2026-07-04T1156_kimi_full.md` tied to
repeatable repository checks. It verifies the `.env`, root `TODO`, and
`docs/internal/TODO.md` ignore rules, confirms that the internal TODO exists in
the ignored location, reruns the direct-header SPDX audit, and reruns the
SC-NeuroCore SNN memory-discipline audit against the shared stimulus directory.

Run it after touching `.gitignore`, `docs/internal/TODO.md`,
`tools/spdx_header_audit.py`, `tools/snn_memory_discipline_audit.py`, or the
SC-NeuroCore SNN stimulus directory:

```bash
PYTHONPATH=src:. python -m pytest tests/test_tools/test_systematic_audit_rerun_contract.py -q
PYTHONPATH=src:. python tools/spdx_header_audit.py --check
PYTHONPATH=src:. python tools/snn_memory_discipline_audit.py \
  --repo . \
  --stimulus-dir "$SC_NEUROCORE_GOTM_ROOT/04_ARCANE_SAPIENCE/snn_stimuli/SC-NEUROCORE" \
  --output docs/internal/snn_memory_discipline_audit.json
```

### Model documentation audit

`tools/audit_model_docs.py` inventories source model modules, documentation
pages, matching tests, and benchmark artifacts. It is a triage tool, not a
scientific approval gate: it can prove that required evidence exists, but model
equations, references, biological interpretation, and numerical fidelity still
need human review before a page is promoted to verified status.

Run the current audit:

```bash
python tools/audit_model_docs.py \
  --repo . \
  --out-dir docs/internal \
  --timestamp "$(date -u +%Y-%m-%dT%H%M%SZ)"
```

Use `--check` only when the repository is expected to have every source model
at `PASS`. During debt burn-down, the generated JSON and Markdown manifests are
the authoritative queue for batching missing tests, benchmark artifacts, and
append-only documentation evidence.

Create a focused review batch without rewriting any model pages:

```bash
python tools/audit_model_docs.py \
  --repo . \
  --out-dir docs/internal \
  --timestamp "$(date -u +%Y-%m-%dT%H%M%SZ)" \
  --batch-status NEEDS_TEST \
  --batch-limit 25
```

Use `--batch-status NEEDS_BENCHMARK` for benchmark-artifact work and
`--batch-status NEEDS_DOC_EVIDENCE` for append-only page evidence work. Treat
these batch files as work queues derived from the full manifest, not as
replacement status records.

Narrow a batch to a specific evidence gap with `--batch-missing`:

```bash
python tools/audit_model_docs.py \
  --repo . \
  --out-dir docs/internal \
  --timestamp "$(date -u +%Y-%m-%dT%H%M%SZ)" \
  --batch-missing has_source_link \
  --batch-limit 25
```

Repeat `--batch-missing` to require multiple missing rubric keys. This is useful
for mechanical cleanup passes, such as source-link evidence, while preserving
the human-review gate for equations and biological interpretation.

### Strict typing and docstring policy

`BROADCAST_2026-06-17_strict_typing_and_docstring_enforcement.md` is enforced
through two committed gates:

- `mypy --strict src/sc_neurocore/` is configured in `pyproject.toml`, run in
  CI, and included in `tools/preflight.py`.
- `pytest tests/test_public_docstring_policy.py -q` validates the audited
  public Python files listed in `docs/docstring_policy.toml`.

The required `python -m tools.docstring_policy_guard` command preserves the
original Git policy cohort and documentation floor. Every added or modified
tracked Python file and every nonignored untracked Python file must be enrolled,
including tests, tooling and source roots outside the main package. Removing a
file while lowering its declared count, reducing the minimum length or adding
missing-symbol allowances fails the guard. Policy sources must exist inside the
repository. Scope acceptance is separate from native documentation acceptance.

Locally the baseline is committed `HEAD`, or the commit named by `--since REV`
or by the `SC_NEUROCORE_POLICY_BASELINE` variable. The variable reaches every
check that reads this baseline, including the Go and Rust documentation
ratchets below. The preflight passes the merge base of `HEAD` with the branch
it pushes to, as `--since` to the guard and as the variable to the hooks it
runs, so a push is checked over the same commits as the hosted push event; CI
refuses both. GitHub Actions uses the push event's
original `before` commit or the pull request's `base.sha`; manual runs use the
current commit's first parent. The checkout must match the triggering commit.
Missing history, unavailable native Git or malformed event inputs fail the
guard. New-ref pushes without an established original baseline need an explicit
baseline migration before qualification. With the pre-commit hooks installed
(`pre-commit install`), the guard runs on every commit; the preflight runs it over
the whole push range; the CI lint job and the public policy tests run the same
guard.

The docstring policy uses Ruff `D` rules with the NumPy-convention pydocstyle
setting, an isolated configuration and disabled inline suppressions. The
maintained file list grows package-by-package as public surfaces
are audited. Add a file to `docs/docstring_policy.toml` only after its public
module, class, function, method, and property docstrings have been reviewed for
accuracy. Keep the scoped policy passing until `D` can be promoted to the global
Ruff select.

Run the gate after touching policy-listed files, public docstrings, Mypy
configuration, or CI/preflight quality commands:

```bash
PYTHONPATH=src:. python -m mypy --strict src/sc_neurocore/
python -m tools.docstring_policy_guard
PYTHONPATH=src:. python -m pytest tests/test_public_docstring_policy.py -q
PYTHONPATH=src:. python -m pytest tests/test_tools/test_strict_typing_docstring_policy.py -q
```

### Go documentation measurement

`python -m tools.go_doc_ratchet` compares tracked and nonignored untracked Go
with its original Git ceiling and individual declaration allowances. Native
Git, Go and parser failures stop the check. The
submitted paths must be unique relative Go paths without control characters,
leading or trailing whitespace, traversal or symbolic links. This prevents the
parser's line-delimited input from substituting a different file. Its summary
must match the submitted file count, schema and finding totals. Protocol v3
includes all retained exported identities and SHA-256 of the exact bytes passed to the parser and distinguishes
methods by receiver. A documented old case cannot excuse a new undocumented
case with the same aggregate count. Original undocumented declarations must
remain; deleting them does not repay documentation debt.

The measured source files, parser and present root Go module/workspace files
are hashed before and after measurement. The Git-discovered cohort and native Go
version must also remain unchanged. The documentation report and `--update`
records include this source manifest and its SHA-256 alongside the exact argv.
The legacy outer `source_sha256` field remains a Git revision; the nested
`measurement.source_sha256` binds observed file bytes. External compiler and
standard-library bytes are outside this capsule. The ratchet independently
parses the original Git sources selected by local HEAD (or
`SC_NEUROCORE_POLICY_BASELINE`) or the actual CI event
and refuses cohort removal, scalar inflation and new individual debt. Historical
source images use the compiler release recorded in their immutable ceiling,
including a version recheck, while current source uses the CI toolchain. Updating
the current compiler therefore retains reproducible original debt evidence. A private
`--update` migrates the allowance to ceiling protocol v2 and may lower it.
Ignored, generated and external-source ownership still require explicit mapping.

The lint and pre-commit CI jobs pin Go 1.27.4. Lint runs the native ratchet,
`go vet`, formatting and strict Python adapter/documentation checks; the native
documentation job, which installs the test dependencies, type-checks the
contract tests, and the normal test matrix exercises the module-specific native
contracts below.

Run the module-specific real-parser contracts after changing this measurement:

```bash
python -m pytest tests/test_tools/test_go_doc_measurement_native.py tests/test_tools/test_go_doc_history_native.py tests/test_tools/test_godoc_coverage_native.py tests/test_tools_go_doc_ratchet.py -q
```

### Rust documentation measurement

`python -m tools.rust_doc_ratchet` protects the engine library's original Git
source cohort, compiler debt and individual declaration identities. The actual
compiler produces Cargo JSON diagnostics with `--force-warn missing_docs`;
warning color, lint attributes and cap-lints cannot turn missing docs into zero.
A source/config digest is passed in the compiler invocation so another source
image cannot share an identical measurement argument set. A separate invocation
identifier forces the selected library to compile again, and its Cargo artifact
must confirm that the compiler ran. A cached artifact cannot supply this proof.
Unrecognized compiler stdout stops measurement on every invocation, including
output from procedural macros; it cannot become an accepted zero through caching.

The locked native `tools/rustdoc_symbols` parser resolves item and member
ownership from Rust syntax, with hashes of the exact parsed UTF-8 bytes. Native
declaration ranges must stay inside those bytes and preserve UTF-8 boundaries.
Relative compiler source paths resolve against the Cargo workspace root in
which the compiler ran, which is the package directory for a standalone package
and the workspace directory for a member such as the engine. A same-name file at
the repository root cannot take a nested package's diagnostic.
Missing or ambiguous compiler spans, compilation failure and changed inputs
stop measurement. Identical field or method names in different types are
separate cases. Original undocumented declarations must remain in the source
cohort; documenting them reduces debt, deleting them does not repay it.
A declaration that exists only after macro expansion has no syntax identity. In
the current source it is refused until the macro documents it. In the original
baseline it counts as debt under an identity built from the macro name, the
invocation text and the definition text; its retention cannot be checked.

The ratchet independently compiles immutable original Git sources, selected
like the Go ratchet's by local HEAD (or `SC_NEUROCORE_POLICY_BASELINE`) or the
actual CI event, and checks
original ceiling/version reproduction. `--update` records individual v2
allowances and may only lower them. The legacy outer `source_sha256` remains a
Git revision; nested source pins are byte hashes. The selected default library
configuration is measured. Other targets, platform cfgs and external
compiler/dependency inputs need their own qualification. The engine ceiling
is kept in the aggregate v1 form; a v2 record of the whole engine is about
10 MB and its adoption is a separate decision.

Every compiler call of a measurement, the two version queries, `cargo metadata`
and the compilation, runs the toolchain the measured crate pins: the nearest
`rust-toolchain.toml` or legacy `rust-toolchain` from the manifest's directory
up to the repository root, never beyond it (for the engine,
`engine/rust-toolchain.toml`). The measurement sets it as `RUSTUP_TOOLCHAIN`
with `RUSTUP_AUTO_INSTALL=0`, over whatever the caller's environment names, and
names the pinned toolchain's own `bin/rustc` in `RUSTC`. It removes
`CARGO_BUILD_RUSTC`, `CARGO_BUILD_RUSTC_WRAPPER` and
`CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER` and sets `RUSTC_WRAPPER` and
`RUSTC_WORKSPACE_WRAPPER` empty, so neither the caller nor `build.rustc` or a
wrapper in a Cargo configuration file can make Cargo run another compiler. Cargo
starts from the repository root, where rustup alone would not find a member
crate's file. The original revision compiles in its own folder outside the
checkout, which no rustup directory override reaches; that revision reads its
own pin from its image. The folder is `sc-neurocore-rust-doc-original` in the
temporary directory, or the one `SC_NEUROCORE_RUST_DOC_ORIGINAL_DIR` names
outside the checkout. On every run, everything in it except Cargo's build
outputs (`target`, `native-target`) and the ratchet's marker file is deleted.
The sources are then extracted afresh, so the override must name a folder of
its own. A non-empty folder without the marker is refused and left whole. The
folder is created for the current account alone (mode 0700). A link, a folder
of another account and one that group or others can write are refused. The
folder is opened without following a link, and its owner and mode are read from
that open folder, so the folder checked is the folder locked and used; an
override that names a link is refused, not followed. The build outputs persist,
so the original's dependencies, Z3 from C++ source among them, are built once
and not on every run; a cold build could outlast
the measurement's own timeout. The folder can hold several gigabytes. It may
be deleted when no run holds it, and the next run is then cold again. A run
holds an exclusive lock on the folder; a second run says on stderr that it
waits for that lock, then takes its turn.
A pinned toolchain that does not run is refused by name and is not
fetched. A numbered pin must be exactly the release that this compiler
executable reports, so a pre-release of it, or a compiler on the path that
ignores rustup, cannot stand in for it. Two
toolchain files in one directory, a path toolchain, a missing or malformed
channel and a file that is not UTF-8 text are refused. The pin file is a
measured input. A crate without a pin is measured with the caller's
toolchain, and the measurement record states the pin or its absence.

The `rust-documentation` CI job pins Rust 1.98.1, runs the native parser's
format/build/Clippy checks, strict adapter checks and real native contracts,
then runs the original engine ratchet. The build job depends on this check;
the Rust pre-commit hook uses the same ratchet.

```bash
python -m pytest tests/test_tools/test_rustdoc_symbols_native.py tests/test_tools/test_rust_doc_measurement_native.py tests/test_tools/test_rust_doc_history_native.py tests/test_tools/test_rust_doc_ratchet_native.py tests/test_tools/test_rust_doc_toolchain_native.py tests/test_tools_rust_doc_ratchet.py -q
```

### Rust model field documentation

`python -m tools.rust_model_field_docs` documents the public fields of the
engine structs that share their name with a neuron model descriptor's
`class_name`. When the descriptor gives a field a `meaning` and the Rust field
has no documentation yet, the tool writes that meaning with its unit as the
doc comment, for example `/// Leak reversal potential (mV).`; unit `1` is
written as dimensionless, and Markdown link and emphasis characters are
escaped. A parameter takes precedence over a state variable of the same name.
Existing documentation is never replaced, so a hand-written comment stays.
Two descriptors of one class, a described struct defined twice, an unclosed
struct, an unreadable descriptor or an empty meaning stop the tool with
status 2 before anything is written. `--check` writes nothing and fails while
any described field lacks documentation.
After writing, measure under the ceiling's Rust release and lower the engine's
aggregate v1 ceiling in `engine/missing_docs_ceiling.json` to the measured figure,
with the measured revision as `source_sha256`. `--update` would migrate the engine
ceiling to the individual v2 record, which is a separate decision.

### SHD Vertex corrected-selection summary

`tools/summarise_shd_vertex_runs.py` aggregates downloaded SHD Vertex run
artifacts after the deployable checkpoint-selection fix. It scores checkpoint
selection under rounded-delay deployable conditions and keeps native-validation
epochs visible, so regressions caused by native-sigma selection remain obvious.

Run the aggregate after downloading completed jobs:

```bash
python tools/summarise_shd_vertex_runs.py \
  --root data/masquelier_shd/cloud_results \
  --out-prefix docs/internal/shd_vertex_corrected_selection_summary_$(date -u +%Y_%m_%d)
```

Before updating external claims or replying with final SHD accuracy numbers,
verify that all intended seeds are present and that the summary includes the
round-each-epoch comparison run when it is available.

### EDA toolchain inventory

`tools/eda_toolchain_versions.py` captures the local hardware-toolchain
evidence context. It records Vivado, OpenROAD, Yosys, nextpnr, IceStorm,
Trellis, Quartus, Lattice Diamond/Radiant, PYNQ, and OpenROAD/PDK pin fields.

Run a local inventory:

```bash
python tools/eda_toolchain_versions.py \
  --pretty \
  --out build/eda-toolchain.json
```

For release evidence, fail fast on required tool versions:

```bash
python tools/eda_toolchain_versions.py \
  --require vivado \
  --expect vivado=2025.2 \
  --pretty \
  --out build/eda-toolchain.json
```

Do not publish OpenROAD area, power, timing, or GDSII claims unless the exact
OpenROAD binary or container digest and PDK revision are attached to the
generated inventory.

### Toolchain manifest

`tools/toolchain_manifest.toml` gives one answer per tool for the hosted runs
and for this host: the runner images, the Python, Rust (stable and a dated
nightly), Go, Node, npm, pixi, Julia and Mojo versions, the Ubuntu packages
whose version shapes a result, the Windows Icarus Verilog build, and the
published HDL SDK archive. `python -m tools.toolchain_manifest` holds the
repository to it:

```bash
python -m tools.toolchain_manifest --check       # workflows, actions and pins agree
python -m tools.toolchain_manifest --check-host  # each tool on PATH answers its version
python -m tools.toolchain_manifest --list        # every declaration and where it stands
```

- **`--check`** reads the workflows, the composite actions,
  `engine/rust-toolchain.toml`, the `go.mod` files, the Mojo `pixi.toml` and
  `requirements-dev.txt`. It refuses:
  - a value that differs from the manifest;
  - a floating spelling: a `*-latest` runner anywhere, an undated `nightly`, or
    an install of a listed tool without its version, in any spelling of
    `apt-get install`, `apt install` (options before or after the verb) or
    `choco install` (`--version=V` or `--version V`);
  - a setup version it cannot read as a literal: an `env`, `vars` or `inputs`
    reference, or a `matrix` key the job's matrix does not give. A
    `${{ matrix.<key> }}` value is read under the key it names;
  - a setup action the manifest does not know, in a workflow or a composite
    action.
- **Python minors.** A Python minor alone (`3.12`) is accepted only in the
  wheel jobs the manifest names. Python's security-only branches publish no
  Windows or macOS binaries for their newer patches.
- **The pre-commit hook.** `toolchain-manifest` runs the check whenever one of
  those files changes, and the Pre-commit workflow runs it on every push to
  main.
- **`--check-host`** runs each tool with the project's `.venv/bin` first on
  `PATH`. It is the host-side counterpart of the check.
  `tools/eda_toolchain_versions.py` above records the toolchain context of one
  measurement and does not compare it with the manifest.
- **The cargo-fuzz runner** (`tools/security_scan/run_cargo_fuzz_scanners.py`)
  reads its dated nightly from the manifest.
- **A version raise** is one edit of the manifest followed by every declaration
  the check then names.

## Validation

Focused tests for these tools live under `tests/test_tools/`.

```bash
pytest tests/test_tools -q
ruff check tools tests/test_tools
ruff format --check tools tests/test_tools
mypy tools/audit_model_docs.py tools/summarise_shd_vertex_runs.py tools/eda_toolchain_versions.py
```

## SHD platform coverage

The main CI workflow runs the public SHD executable-reader refusal on an actual
Windows runner. Linux runs the recording and worker tests, including watchdog
exits. The primary coverage job combines both native platform records using
relative source paths before enforcing the unchanged 100% aggregate gate.
The Windows refusal tests skip on Linux and do not replace the OS identity.
