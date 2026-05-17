<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->

# Network Formal Verification

`sc-neurocore formal verify-network` emits a deterministic dense LIF fixture,
SystemVerilog assertions, a SymbiYosys job file, and a validated JSON report for
network-level rate and refractory checks. The command is designed for CI and
release evidence: invalid contracts fail before artefacts are accepted, and
counterexample traces are replayed against the same property parameters recorded
in the report.

```bash
sc-neurocore formal verify-network \
  --module-name dense_lif_frontier_fixture \
  --input-width 3 \
  --output-width 2 \
  --state-width 16 \
  --output-index 0 \
  --window-cycles 8 \
  --max-spikes 4 \
  --refractory-cycles 2 \
  --output build/formal
```

The default report path is
`build/formal/formal_rate_bound_report.json`. Supplying `--out` writes the same
schema to an explicit path.

## Generated Artefacts

For a module named `dense_lif_frontier_fixture`, the command writes:

| Artefact | Purpose |
|----------|---------|
| `dense_lif_frontier_fixture.v` | Deterministic dense LIF fixture RTL |
| `dense_lif_frontier_fixture_rate_bound.sv` | Rate-bound assertion module and executable `bind` |
| `dense_lif_frontier_fixture_refractory.sv` | Refractory assertion module and executable `bind`, when `--refractory-cycles` is set |
| `dense_lif_frontier_fixture_formal_bundle.sv` | Combined assertion bundle used by Yosys/SymbiYosys |
| `dense_lif_frontier_fixture.sby` | Bounded model checking job file |
| `formal_rate_bound_report.json` | Machine-readable verification report |

`--run-symbiyosys` executes `sby -f <module>.sby` when `sby` is available on
`PATH`. If SymbiYosys is not installed, the command still writes the generated
artefacts and records `tool_unavailable` in the report instead of fabricating a
proof result. A failed proof or replay violation returns a non-zero exit code.

## Report Contract

Reports are versioned by `FORMAL_NETWORK_REPORT_SCHEMA_VERSION` and validated by
`validate_formal_network_report` before they are written. The stable top-level
fields are:

| Field | Meaning |
|-------|---------|
| `schema_version` | Formal report schema identifier |
| `network` | Dense LIF fixture dimensions, signal names, and module name |
| `rate_bound` | Output index, window length, and max spike budget |
| `refractory` | Optional output index and refractory-cycle contract |
| `artifacts.rtl` | Generated RTL fixture path |
| `artifacts.sva` | Compatibility alias for the rate-bound assertion path |
| `artifacts.rate_sva` | Generated rate-bound assertion path |
| `artifacts.refractory_sva` | Generated refractory assertion path or `null` |
| `artifacts.formal_bundle` | Combined assertion bundle path |
| `artifacts.sby` | SymbiYosys job path |
| `artifacts.report` | Report path |
| `rate_replay` | Counterexample replay result for the rate-bound contract |
| `replay` | Backward-compatible alias of `rate_replay` |
| `refractory_replay` | Counterexample replay result for the refractory contract |
| `symbiyosys` | Tool status, return code, stdout, and stderr |

The validator rejects unknown schema versions, invalid network dimensions,
invalid property parameters, mismatched replay aliases, mismatched `.sby`
references, path traversal outside an optional artefact root, and missing
artefact files when filesystem validation is requested.

```python
import json
from pathlib import Path

from sc_neurocore.formal import validate_formal_network_report

report_path = Path("build/formal/formal_rate_bound_report.json")
payload = json.loads(report_path.read_text(encoding="utf-8"))
validate_formal_network_report(payload, artifact_root=report_path.parent)
```

## Python API

The CLI is backed by the same public formal helpers exposed from
`sc_neurocore.formal`:

```python
from sc_neurocore.formal import (
    DenseLIFNetworkSpec,
    NetworkRateBound,
    NetworkRefractoryInvariant,
    compile_dense_lif_fixture_rtl,
    compile_network_rate_bound_sva,
    compile_network_refractory_sva,
    validate_formal_network_report,
)
```

`compile_dense_lif_fixture_rtl` emits the deterministic fixture RTL.
`compile_network_rate_bound_sva` and `compile_network_refractory_sva` emit
Yosys-compatible SystemVerilog assertion modules with executable `bind`
statements. The report validator is intentionally independent of the CLI so CI
jobs and downstream release tooling can validate archived evidence without
regenerating artefacts.

## CI Evidence Check

Repository CI jobs can use `tools/verify_formal_network_evidence.py` as the
single formal-evidence gate. The tool runs `sc-neurocore formal verify-network`,
passes `--run-symbiyosys` only when `sby` is available, then reloads
`formal_rate_bound_report.json` and validates it with
`validate_formal_network_report(..., artifact_root=<output>)`.

```bash
PYTHONPATH=src python tools/verify_formal_network_evidence.py \
  --output build/formal-evidence \
  --summary build/formal-evidence/summary.json
```

The summary JSON records the command, return code, captured output, artefact
root, report path, whether SymbiYosys was requested, and the final pass/fail
state. A missing `sby` binary is not treated as a fabricated proof; it simply
keeps SymbiYosys unrequested while still enforcing generated artefact and report
schema validity.

## Scope

The current command verifies generated properties over a deterministic dense LIF
fixture. It does not claim whole-model biological correctness and does not
replace model-specific validation, simulation parity, or hardware synthesis
evidence. Its purpose is a strict, reproducible formal-evidence surface for
network-level spike-rate and refractory invariants.
