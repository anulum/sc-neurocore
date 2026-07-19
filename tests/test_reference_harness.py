# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reference-trace harness contracts

"""Production contracts for the neuron reference-trace validation harness."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from sc_neurocore.neurons.reference_traces import (
    ReferenceTraceSpec,
    list_reference_trace_specs,
    load_reference_trace_spec,
    reference_trace_spec_from_payload,
    simulate_reference_trace,
    validate_all_reference_traces,
    validate_reference_trace,
    validate_reference_trace_spec,
)
from sc_neurocore.neurons.universal_dsl import list_bundled_schemas

_SUPPORTED_REFERENCE_RUNNERS = frozenset({"universal_dsl", "hand_model", "hand_and_universal_dsl"})
_EXTERNAL_SOURCE_PREFIXES = {
    "coba_lif": "Brette et al. (2007)",
    "iqif": "https://github.com/twetto/iq-neuron/blob/",
}
_DETERMINISTIC_SCHEMA_TRACES = {
    "adaptive_threshold_if": "adaptive_threshold_if_tonic_adaptation_doi",
    "adex": "adex_resting_adaptation_doi",
    "cazelles_map": "cazelles_map_bursting_doi",
    "chialvo_map": "chialvo_map_doi",
    "coba_lif": "coba_lif_conductance_rk4_doi",
    "connor_stevens": "connor_stevens_driven_spiking_doi",
    "courage_nekorkin_map": "courage_nekorkin_map_autonomous_doi",
    "dpi_neuron": "dpi_neuron_driven_spiking_doi",
    "ermentrout_kopell_map_neuron": "ermentrout_kopell_theta_euler_doi",
    "exp_if": "exp_if_driven_rk4_doi",
    "fitzhugh_nagumo": "fitzhugh_nagumo_driven_oscillation_doi",
    "fitzhugh_rinzel": "fitzhugh_rinzel_driven_bursting_doi",
    "glif": "glif_constant_current_threshold_adaptation",
    "hindmarsh_rose": "hindmarsh_rose_short_bursting_prefix",
    "hodgkin_huxley": "hodgkin_huxley_driven_spiking_doi",
    "izhikevich": "izhikevich_regular_spiking_doi",
    "izhikevich2007": "izhikevich2007_regular_spiking_doi",
    "ibarz_tanaka_map": "ibarz_tanaka_map_2007_doi",
    "iqif": "iqif_a8752eb_tutorial",
    "lapicque": "lapicque_constant_current_closed_form",
    "lif": "lif_constant_current_closed_form",
    "mckean": "mckean_driven_oscillation_doi",
    "mcculloch_pitts": "mcculloch_pitts_1943_truth_table",
    "medvedev_map": "medvedev_map_first_return_doi",
    "mihalas_niebur": "mihalas_niebur_driven_spiking_doi",
    "morris_lecar": "morris_lecar_driven_oscillation_doi",
    "pernarowski": "pernarowski_autonomous_bursting_doi",
    "terman_wang": "terman_wang_legion_oscillation_doi",
    "wilson_hr": "wilson_hr_driven_spiking_doi",
    "perfect_integrator": "perfect_integrator_constant_current_sawtooth",
    "quadratic_if": "quadratic_if_zero_current_analytic",
    "resonate_fire": "resonate_fire_subthreshold_resonance_doi",
    "rulkov_map": "rulkov_map_driven_spiking_doi",
    "sigmoid_rate": "sigmoid_rate_exact_relaxation_doi",
    "theta": "theta_constant_current_phase_analytic",
    "threshold_linear_rate": "threshold_linear_rate_rectifier_doi",
    "wang_buzsaki": "wang_buzsaki_driven_spiking_doi",
}


def _committed_reference_routes() -> dict[str, tuple[str, str]]:
    """Return artefact name to schema/runner routes from committed JSON."""
    data_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/reference_trace_data"
    routes: dict[str, tuple[str, str]] = {}
    for path in sorted(data_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload["model"]
        routes[payload["name"]] = (model["schema_name"], model["runner"])
    return routes


def test_seeded_corpus_has_analytic_schema_entries() -> None:
    """The seed corpus must expose deterministic analytic schema references."""
    names = list_reference_trace_specs()

    assert names == tuple(sorted(names))
    assert set(_DETERMINISTIC_SCHEMA_TRACES.values()) <= set(names)
    assert "escape_rate_lfsr16_statistical_v1" not in names
    assert "poisson_lfsr16_statistical_v1" not in names
    assert "wong_wang_appendix_euler_ou_doi" not in names

    spec = load_reference_trace_spec("lif_constant_current_closed_form")
    assert isinstance(spec, ReferenceTraceSpec)
    assert spec.schema_name == "lif"
    assert spec.provenance.kind == "analytic_closed_form"
    assert spec.protocol.state_variables == ("v",)
    assert spec.protocol.inputs["I"] == 1.0


def test_reference_trace_corpus_dispatches_every_artefact_by_runner_class() -> None:
    """Every committed artefact uses a supported generic or dedicated route."""
    routes = _committed_reference_routes()
    runners = {runner for _, runner in routes.values()}
    generic_routes = {
        schema_name: trace_name
        for trace_name, (schema_name, runner) in routes.items()
        if runner == "universal_dsl"
    }
    dedicated_trace_names = {
        trace_name for trace_name, (_, runner) in routes.items() if runner != "universal_dsl"
    }

    assert runners <= _SUPPORTED_REFERENCE_RUNNERS
    assert generic_routes == _DETERMINISTIC_SCHEMA_TRACES
    assert set(list_reference_trace_specs()) == set(generic_routes.values())
    assert dedicated_trace_names.isdisjoint(list_reference_trace_specs())
    assert {schema_name for schema_name, _ in routes.values()} <= set(list_bundled_schemas())

    for schema_name, trace_name in _DETERMINISTIC_SCHEMA_TRACES.items():
        spec = load_reference_trace_spec(trace_name)
        assert spec.schema_name == schema_name
        assert spec.runner == "universal_dsl"
        external_prefix = _EXTERNAL_SOURCE_PREFIXES.get(schema_name)
        if external_prefix is None:
            assert spec.provenance.source.endswith(f"/{schema_name}.toml")
        else:
            assert spec.provenance.source.startswith(external_prefix)
        assert spec.provenance.citation is not None
        assert spec.provenance.citation
        if "doi" in trace_name:
            assert spec.provenance.citation.startswith("doi:")


def test_simulation_exercises_universal_schema_runner() -> None:
    """The harness must execute the committed schema through UniversalNeuron."""
    spec = load_reference_trace_spec("lapicque_constant_current_closed_form")

    simulation = simulate_reference_trace(spec)

    assert simulation.name == spec.name
    assert simulation.steps == spec.protocol.steps
    assert tuple(simulation.trace) == ("v",)
    assert len(simulation.trace["v"]) == spec.protocol.steps
    assert simulation.trace["v"][0] > 0.0
    assert simulation.spikes == tuple(0 for _ in range(spec.protocol.steps))
    assert simulation.features["spike_count"] == 0.0
    assert simulation.features["first_spike_step"] == -1.0
    assert simulation.features["max.v"] == max(simulation.trace["v"])


def test_stateless_reference_trace_records_event_features_only() -> None:
    """The corpus can validate logical activity without fake membrane state."""
    artifact = (
        Path(__file__).resolve().parents[1]
        / "src/sc_neurocore/neurons/reference_trace_data/mcculloch_pitts_1943_truth_table.json"
    )
    spec = reference_trace_spec_from_payload(json.loads(artifact.read_text(encoding="utf-8")))
    simulation = simulate_reference_trace(spec)

    assert dict(simulation.trace) == {}
    assert simulation.spikes == (1, 1, 1, 1)
    assert simulation.features == {"spike_count": 4.0, "first_spike_step": 1.0}


def test_reference_trace_validation_accepts_seeded_corpus() -> None:
    """All committed seed references must pass their own tolerance contracts."""
    reports = validate_all_reference_traces()

    assert {report.name for report in reports} == set(list_reference_trace_specs())
    assert all(report.passed for report in reports)
    assert all(report.mismatches == () for report in reports)


def test_name_based_simulation_and_validation_paths_are_public() -> None:
    """Name-based public helpers must route through the committed corpus."""
    simulation = simulate_reference_trace("lif_constant_current_closed_form")
    report = validate_reference_trace("lif_constant_current_closed_form")

    assert simulation.name == "lif_constant_current_closed_form"
    assert report.name == simulation.name
    assert report.passed


def test_unknown_reference_trace_name_fails_closed() -> None:
    """Unknown corpus identifiers must not silently fall back to another trace."""
    with pytest.raises(ValueError, match="unknown reference trace"):
        load_reference_trace_spec("not_a_committed_reference")


def test_validation_reports_feature_drift() -> None:
    """A drifted expected feature must fail closed with the feature name."""
    spec = load_reference_trace_spec("lif_constant_current_closed_form")
    drifted_features = dict(spec.expected_features)
    drifted_features["final.v"] += 0.25
    drifted = replace(spec, expected_features=drifted_features)

    report = validate_reference_trace_spec(drifted)

    assert not report.passed
    assert [mismatch.feature for mismatch in report.mismatches] == ["final.v"]


def test_simulation_rejects_unsupported_in_memory_runner() -> None:
    """In-memory specs cannot select runners outside the v1 production surface."""
    spec = load_reference_trace_spec("lif_constant_current_closed_form")
    unsupported = replace(spec, runner="python_loop")

    with pytest.raises(ValueError, match="unsupported reference-trace runner"):
        simulate_reference_trace(unsupported)
