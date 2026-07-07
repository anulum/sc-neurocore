<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Reference Trace Harness

The reference-trace harness validates schema-driven neuron models against
committed scalar feature contracts. A corpus entry defines the schema model,
runner, deterministic protocol, provenance, expected features, and per-feature
tolerances. The production validator loads those JSON entries from the package,
executes the same `UniversalNeuron` runner used by public schema workflows, and
reports feature-level mismatches without falling back to another trace.

This page documents the WC-A1 seed and early expansion corpus. It does not
claim that the full external NEST, Brian2, NEURON, or published-figure corpus
is complete.

## Current Corpus

The committed entries are analytic references for deterministic schema models:

| Trace | Schema | Runner | Provenance |
|-------|--------|--------|------------|
| `lif_constant_current_closed_form` | `lif` | `universal_dsl` | Closed-form RC solution from `neurons/model_schemas/lif.toml` |
| `lapicque_constant_current_closed_form` | `lapicque` | `universal_dsl` | Closed-form RC solution from `neurons/model_schemas/lapicque.toml` |
| `perfect_integrator_constant_current_sawtooth` | `perfect_integrator` | `universal_dsl` | Analytic post-reset sawtooth solution from `neurons/model_schemas/perfect_integrator.toml` |
| `quadratic_if_zero_current_analytic` | `quadratic_if` | `universal_dsl` | Analytic zero-current Riccati solution from `neurons/model_schemas/quadratic_if.toml` with DOI-backed schema provenance |

All entries record the membrane variable `v`, spike count, first spike step,
and final/min/max/mean voltage features. The tests independently recompute the
LIF, QIF, and perfect-integrator analytic solutions so the committed feature
values are not merely copied from the runner output. The perfect-integrator
entry is intentionally spike-bearing; it validates the schema runner's
post-threshold reset state and first-spike feature, not only a quiet voltage
trajectory. The QIF tolerance is intentionally wider than the exact-feature
precision because the current schema runner declares explicit Euler integration
while the reference is the continuous zero-current solution.

## Public API

```python
from sc_neurocore.neurons.reference_traces import validate_all_reference_traces

reports = validate_all_reference_traces()
assert all(report.passed for report in reports)
```

Use `validate_reference_trace(name)` for one committed trace, or
`reference_trace_spec_from_payload(payload)` when reviewing a candidate corpus
entry before committing it. Malformed payloads fail closed on schema version,
runner, schema name, protocol fields, feature values, and tolerance fields.

## Verification

The focused harness selector is:

```bash
PYTHONPATH=src python -m pytest tests/test_reference_traces.py tests/test_reference_trace_payloads.py -q
```

Exact-file coverage for the implementation modules is measured with:

```bash
PYTHONPATH=src python -m coverage run --rcfile=/dev/null --source=src/sc_neurocore/neurons -m pytest tests/test_reference_traces.py tests/test_reference_trace_payloads.py -q
PYTHONPATH=src python -m coverage report --rcfile=/dev/null --include='src/sc_neurocore/neurons/reference_trace*.py' --fail-under=100 -m
```

## Remaining WC-A1 Work

The next corpus expansion should add external or simulator-backed references for
the remaining deterministic schema-DSL models. Each new entry should carry
source-level provenance, independent feature derivation where possible, and a
focused parity test before the wider WC-A1 item is closed.
