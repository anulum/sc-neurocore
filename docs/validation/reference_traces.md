<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Reference Trace Harness

The reference-trace harness validates schema-driven neuron models against
committed scalar feature contracts. A corpus entry defines the schema model,
runner, deterministic protocol, provenance, expected features, and per-feature
tolerances. The production validator loads those JSON entries from the package,
executes the same `UniversalNeuron` runner used by public schema workflows, and
reports feature-level mismatches without falling back to another trace.

This page documents the WC-A1 deterministic schema corpus. It does not claim
NEST, Brian2, NEURON, or published-figure replay coverage; those remain separate
external-simulator validation surfaces.

## Current Corpus

The committed corpus has one reference entry for every deterministic bundled
schema model. `poisson` and `escape_rate` are excluded from this deterministic
table because their schemas are stochastic.

| Trace | Schema | Runner | Provenance |
|-------|--------|--------|------------|
| `adex_resting_adaptation_doi` | `adex` | `universal_dsl` | Independent explicit-Euler re-derivation of the subthreshold equations from `neurons/model_schemas/adex.toml` with DOI-backed schema provenance |
| `connor_stevens_resting_gate_doi` | `connor_stevens` | `universal_dsl` | DOI-backed resting gate prefix from `neurons/model_schemas/connor_stevens.toml` |
| `exp_if_resting_exponential_doi` | `exp_if` | `universal_dsl` | Independent explicit-Euler re-derivation of the resting equation from `neurons/model_schemas/exp_if.toml` with DOI-backed schema provenance |
| `fitzhugh_nagumo_driven_oscillation_doi` | `fitzhugh_nagumo` | `universal_dsl` | Independent explicit-Euler re-derivation of the driven relaxation equations from `neurons/model_schemas/fitzhugh_nagumo.toml` with DOI-backed schema provenance |
| `glif_constant_current_threshold_adaptation` | `glif` | `universal_dsl` | Analytic linear Euler recurrence from `neurons/model_schemas/glif.toml` with DOI-backed schema provenance |
| `hindmarsh_rose_short_bursting_prefix` | `hindmarsh_rose` | `universal_dsl` | DOI-backed short finite prefix from `neurons/model_schemas/hindmarsh_rose.toml` |
| `hodgkin_huxley_resting_gate_doi` | `hodgkin_huxley` | `universal_dsl` | DOI-backed resting conductance-gate prefix from `neurons/model_schemas/hodgkin_huxley.toml` |
| `izhikevich_regular_spiking_doi` | `izhikevich` | `universal_dsl` | Independent explicit-Euler re-derivation of the regular-spiking equations from `neurons/model_schemas/izhikevich.toml` with DOI-backed schema provenance |
| `lif_constant_current_closed_form` | `lif` | `universal_dsl` | Closed-form RC solution from `neurons/model_schemas/lif.toml` |
| `lapicque_constant_current_closed_form` | `lapicque` | `universal_dsl` | Closed-form RC solution from `neurons/model_schemas/lapicque.toml` |
| `morris_lecar_depolarizing_current_doi` | `morris_lecar` | `universal_dsl` | DOI-backed depolarizing calcium-potassium trace from `neurons/model_schemas/morris_lecar.toml` |
| `perfect_integrator_constant_current_sawtooth` | `perfect_integrator` | `universal_dsl` | Analytic post-reset sawtooth solution from `neurons/model_schemas/perfect_integrator.toml` |
| `quadratic_if_zero_current_analytic` | `quadratic_if` | `universal_dsl` | Analytic zero-current Riccati solution from `neurons/model_schemas/quadratic_if.toml` with DOI-backed schema provenance |
| `resonate_fire_subthreshold_resonance_doi` | `resonate_fire` | `universal_dsl` | Analytic linear Euler recurrence from `neurons/model_schemas/resonate_fire.toml` |
| `rulkov_map_short_window_boundary` | `rulkov_map` | `universal_dsl` | DOI-backed short finite boundary trace from `neurons/model_schemas/rulkov_map.toml` |
| `theta_constant_current_phase_analytic` | `theta` | `universal_dsl` | Analytic tangent half-angle phase solution from `neurons/model_schemas/theta.toml` with DOI-backed schema provenance |
| `wang_buzsaki_resting_interneuron_doi` | `wang_buzsaki` | `universal_dsl` | DOI-backed resting interneuron conductance-gate prefix from `neurons/model_schemas/wang_buzsaki.toml` |

All entries record spike count, first spike step, and final/min/max/mean
features for the declared state variables. The tests independently recompute the
LIF, QIF, perfect-integrator, resonate-fire, theta, GLIF, Izhikevich,
FitzHugh-Nagumo, AdEx, and exponential-IF analytic or explicit-Euler solutions so
the committed feature values for those entries are not merely copied from the
runner output. The GLIF
entry re-derives the exact subthreshold explicit-Euler recurrence for its linear
membrane, adaptive threshold, and two after-spike currents; the Izhikevich entry
re-derives the exact regular-spiking explicit-Euler recurrence including its
`v = c`, `u = u + d` reset; and the FitzHugh-Nagumo entry re-derives its cubic
relaxation recurrence with the `v = -1` reset. The perfect-integrator,
FitzHugh-Nagumo, and Izhikevich entries are spike-bearing;
they validate reset and first-spike features, not only quiet trajectories. The
Rulkov entry is intentionally short-window because the current generic schema
runner executes its map equation through an Euler-style path that diverges on long
windows. The QIF and theta tolerances are wider than
machine-epsilon feature precision because the current schema runner declares
explicit Euler integration while those references are continuous analytic
solutions.

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

## External Simulator Boundary

The deterministic bundled-schema corpus is complete for package-local
`UniversalNeuron` validation. External NEST, Brian2, NEURON, and
published-figure replay traces require separate adapters or recorded fixtures
before they can be represented as external-simulator evidence.
