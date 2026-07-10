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
| `connor_stevens_resting_gate_doi` | `connor_stevens` | `universal_dsl` | Independent explicit-Euler re-derivation of the resting gate equations from `neurons/model_schemas/connor_stevens.toml` with DOI-backed schema provenance |
| `dpi_neuron_driven_spiking_doi` | `dpi_neuron` | `universal_dsl` | Independent explicit-Euler re-derivation of the current-mode differential-pair-integrator membrane from `neurons/model_schemas/dpi_neuron.toml` with DOI-backed schema provenance |
| `exp_if_resting_exponential_doi` | `exp_if` | `universal_dsl` | Independent explicit-Euler re-derivation of the resting equation from `neurons/model_schemas/exp_if.toml` with DOI-backed schema provenance |
| `fitzhugh_nagumo_driven_oscillation_doi` | `fitzhugh_nagumo` | `universal_dsl` | Independent explicit-Euler re-derivation of the driven relaxation equations from `neurons/model_schemas/fitzhugh_nagumo.toml` with DOI-backed schema provenance |
| `glif_constant_current_threshold_adaptation` | `glif` | `universal_dsl` | Analytic linear Euler recurrence from `neurons/model_schemas/glif.toml` with DOI-backed schema provenance |
| `hindmarsh_rose_short_bursting_prefix` | `hindmarsh_rose` | `universal_dsl` | Independent explicit-Euler re-derivation of the short bursting prefix from `neurons/model_schemas/hindmarsh_rose.toml` with DOI-backed schema provenance |
| `hodgkin_huxley_resting_gate_doi` | `hodgkin_huxley` | `universal_dsl` | Independent explicit-Euler re-derivation of the resting gate equations from `neurons/model_schemas/hodgkin_huxley.toml` with DOI-backed schema provenance |
| `izhikevich_regular_spiking_doi` | `izhikevich` | `universal_dsl` | Independent explicit-Euler re-derivation of the regular-spiking equations from `neurons/model_schemas/izhikevich.toml` with DOI-backed schema provenance |
| `izhikevich2007_regular_spiking_doi` | `izhikevich2007` | `universal_dsl` | Independent explicit-Euler re-derivation of the biophysical quadratic equations from `neurons/model_schemas/izhikevich2007.toml` with DOI-backed schema provenance |
| `lif_constant_current_closed_form` | `lif` | `universal_dsl` | Closed-form RC solution from `neurons/model_schemas/lif.toml` |
| `lapicque_constant_current_closed_form` | `lapicque` | `universal_dsl` | Closed-form RC solution from `neurons/model_schemas/lapicque.toml` |
| `morris_lecar_depolarizing_current_doi` | `morris_lecar` | `universal_dsl` | Independent explicit-Euler re-derivation of the depolarizing equations from `neurons/model_schemas/morris_lecar.toml` with DOI-backed schema provenance |
| `perfect_integrator_constant_current_sawtooth` | `perfect_integrator` | `universal_dsl` | Analytic post-reset sawtooth solution from `neurons/model_schemas/perfect_integrator.toml` |
| `quadratic_if_zero_current_analytic` | `quadratic_if` | `universal_dsl` | Analytic zero-current Riccati solution from `neurons/model_schemas/quadratic_if.toml` with DOI-backed schema provenance |
| `resonate_fire_subthreshold_resonance_doi` | `resonate_fire` | `universal_dsl` | Analytic linear Euler recurrence from `neurons/model_schemas/resonate_fire.toml` |
| `rulkov_map_driven_spiking_doi` | `rulkov_map` | `universal_dsl` | Independent piecewise-map iteration (Rulkov 2002, `method="map"`) from `neurons/model_schemas/rulkov_map.toml` with DOI-backed schema provenance |
| `theta_constant_current_phase_analytic` | `theta` | `universal_dsl` | Analytic tangent half-angle phase solution from `neurons/model_schemas/theta.toml` with DOI-backed schema provenance |
| `wang_buzsaki_resting_interneuron_doi` | `wang_buzsaki` | `universal_dsl` | Independent explicit-Euler re-derivation of the resting gate equations from `neurons/model_schemas/wang_buzsaki.toml` with DOI-backed schema provenance |

All entries record spike count, first spike step, and final/min/max/mean
features for the declared state variables. The tests independently recompute the
LIF, QIF, perfect-integrator, resonate-fire, theta, GLIF, Izhikevich,
Izhikevich 2007, FitzHugh-Nagumo, AdEx, exponential-IF, Hindmarsh-Rose, Morris-Lecar,
Hodgkin-Huxley, Connor-Stevens, Wang-Buzsaki, and DPI analytic or explicit-Euler
solutions — every deterministic bundled-schema entry — so the committed feature
values are not merely copied from the runner output. The Morris-Lecar,
Hodgkin-Huxley, Connor-Stevens, and Wang-Buzsaki re-derivations reuse the runner's
numpy activation, exponential, and exprel functions so the conductance rate terms
match bit-for-bit. The GLIF
entry re-derives the exact subthreshold explicit-Euler recurrence for its linear
membrane, adaptive threshold, and two after-spike currents; the Izhikevich entry
re-derives the exact regular-spiking explicit-Euler recurrence including its
`v = c`, `u = u + d` reset; and the FitzHugh-Nagumo entry re-derives its cubic
relaxation recurrence with the `v = -1` reset; and the DPI entry re-derives its
current-mode leaky-integrator recurrence with the `i_mem = i_reset` reset, its
non-negative drive keeping the source model's `max(i_mem, 0)` rectification inert.
The perfect-integrator, FitzHugh-Nagumo, Izhikevich, Izhikevich 2007, and DPI
entries are spike-bearing;
they validate reset and first-spike features, not only quiet trajectories. The
Rulkov entry iterates the Rulkov 2002 piecewise fast/slow map with the
`method = "map"` integration mode (`x_{n+1} = f(x_n, y_n)`, iterated as a discrete
map rather than integrated as an ODE), so the trajectory is bounded and its
committed features are independently re-derived exactly; a driving current
exercises all three fast-map branches (rational subthreshold, spike plateau, hard
reset). The QIF and theta tolerances are wider than
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
