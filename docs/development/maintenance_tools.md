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
`pytest.importorskip(...)` optional dependency gate. Any other uncollected
tracked test file is a failure.

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

Locally the baseline is committed `HEAD`. GitHub Actions uses the push event's
original `before` commit or the pull request's `base.sha`; manual runs use the
current commit's first parent. The checkout must match the triggering commit.
Missing history, unavailable native Git or malformed event inputs fail the
guard. New-ref pushes without an established original baseline need an explicit
baseline migration before qualification. The local hook runs on every commit;
the CI lint job and the public policy tests run the same guard.

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
parses the original Git sources selected by local HEAD or the actual CI event
and refuses cohort removal, scalar inflation and new individual debt. A private
`--update` migrates the allowance to ceiling protocol v2 and may lower it.
Ignored, generated and external-source ownership still require explicit mapping.

The lint and pre-commit CI jobs pin Go 1.27.1. Lint runs the native ratchet,
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

The ratchet independently compiles immutable original Git sources and checks
original ceiling/version reproduction. `--update` records individual v2
allowances and may only lower them. The legacy outer `source_sha256` remains a
Git revision; nested source pins are byte hashes. The selected default library
configuration is measured. Other targets, platform cfgs and external
compiler/dependency inputs need their own qualification. The engine ceiling
is kept in the aggregate v1 form; a v2 record of the whole engine is about
10 MB and its adoption is a separate decision.

The `rust-documentation` CI job pins Rust 1.98.1, runs the native parser's
format/build/Clippy checks, strict adapter checks and real native contracts,
then runs the original engine ratchet. The build job depends on this check;
the Rust pre-commit hook uses the same ratchet.

```bash
python -m pytest tests/test_tools/test_rustdoc_symbols_native.py tests/test_tools/test_rust_doc_measurement_native.py tests/test_tools/test_rust_doc_history_native.py tests/test_tools/test_rust_doc_ratchet_native.py tests/test_tools_rust_doc_ratchet.py -q
```

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

## Validation

Focused tests for these tools live under `tests/test_tools/`.

```bash
pytest tests/test_tools -q
ruff check tools tests/test_tools
ruff format --check tools tests/test_tools
mypy tools/audit_model_docs.py tools/summarise_shd_vertex_runs.py tools/eda_toolchain_versions.py
```
