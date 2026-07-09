# Maintenance Tools

This page records repository maintenance tools that produce audit evidence but
do not change runtime behaviour. Each tool emits timestamped artefacts where
possible, so historical audits remain reproducible.

## 2026-04-30 Tooling Baseline

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
