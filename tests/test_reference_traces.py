# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reference-trace neuron validation contracts

"""Production contracts for the neuron reference-trace validation harness."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from sc_neurocore.neurons.reference_traces import (
    ReferenceTraceSpec,
    list_reference_trace_specs,
    load_reference_trace_spec,
    simulate_reference_trace,
    validate_all_reference_traces,
    validate_reference_trace,
    validate_reference_trace_spec,
)
from sc_neurocore.neurons.universal_dsl import list_bundled_schemas
from tests.cosim_support import (
    _adex_subthreshold_euler_features,
    _closed_form_features,
    _connor_stevens_macrostep_rk4_features,
    _exp_if_subthreshold_euler_features,
    _fitzhugh_nagumo_rk4_features,
    _fitzhugh_rinzel_rk4_features,
    _glif_driven_rk4_features,
    _hindmarsh_rose_prefix_euler_features,
    _hodgkin_huxley_macrostep_rk4_features,
    _izhikevich_rs_euler_features,
    _mckean_rk4_features,
    _morris_lecar_rk4_features,
    _pernarowski_rk4_features,
    _quadratic_if_zero_current_features,
    _resonate_fire_linear_euler_features,
    _rulkov_map_features,
    _terman_wang_rk4_features,
    _theta_constant_current_features,
    _wang_buzsaki_macrostep_gauss_seidel_features,
    _wilson_hr_rk4_features,
)

_STOCHASTIC_SCHEMA_NAMES = frozenset({"escape_rate", "poisson"})
_DETERMINISTIC_SCHEMA_TRACES = {
    "adex": "adex_resting_adaptation_doi",
    "cazelles_map": "cazelles_map_bursting_doi",
    "connor_stevens": "connor_stevens_driven_spiking_doi",
    "dpi_neuron": "dpi_neuron_driven_spiking_doi",
    "exp_if": "exp_if_resting_exponential_doi",
    "fitzhugh_nagumo": "fitzhugh_nagumo_driven_oscillation_doi",
    "fitzhugh_rinzel": "fitzhugh_rinzel_driven_bursting_doi",
    "glif": "glif_constant_current_threshold_adaptation",
    "hindmarsh_rose": "hindmarsh_rose_short_bursting_prefix",
    "hodgkin_huxley": "hodgkin_huxley_driven_spiking_doi",
    "izhikevich": "izhikevich_regular_spiking_doi",
    "izhikevich2007": "izhikevich2007_regular_spiking_doi",
    "lapicque": "lapicque_constant_current_closed_form",
    "lif": "lif_constant_current_closed_form",
    "mckean": "mckean_driven_oscillation_doi",
    "mihalas_niebur": "mihalas_niebur_driven_spiking_doi",
    "morris_lecar": "morris_lecar_driven_oscillation_doi",
    "pernarowski": "pernarowski_autonomous_bursting_doi",
    "terman_wang": "terman_wang_legion_oscillation_doi",
    "wilson_hr": "wilson_hr_driven_spiking_doi",
    "perfect_integrator": "perfect_integrator_constant_current_sawtooth",
    "quadratic_if": "quadratic_if_zero_current_analytic",
    "resonate_fire": "resonate_fire_subthreshold_resonance_doi",
    "rulkov_map": "rulkov_map_driven_spiking_doi",
    "theta": "theta_constant_current_phase_analytic",
    "wang_buzsaki": "wang_buzsaki_driven_spiking_doi",
}


def test_seeded_corpus_has_analytic_schema_entries() -> None:
    """The seed corpus must expose deterministic analytic schema references."""
    names = list_reference_trace_specs()

    assert names == tuple(sorted(names))
    assert set(_DETERMINISTIC_SCHEMA_TRACES.values()) <= set(names)

    spec = load_reference_trace_spec("lif_constant_current_closed_form")
    assert isinstance(spec, ReferenceTraceSpec)
    assert spec.schema_name == "lif"
    assert spec.provenance.kind == "analytic_closed_form"
    assert spec.protocol.state_variables == ("v",)
    assert spec.protocol.inputs["I"] == 1.0


def test_reference_trace_corpus_covers_every_deterministic_bundled_schema() -> None:
    """Every deterministic bundled schema must have one committed trace."""
    deterministic_schemas = set(list_bundled_schemas()) - _STOCHASTIC_SCHEMA_NAMES

    assert set(_DETERMINISTIC_SCHEMA_TRACES) == deterministic_schemas
    for schema_name, trace_name in _DETERMINISTIC_SCHEMA_TRACES.items():
        spec = load_reference_trace_spec(trace_name)
        assert spec.schema_name == schema_name
        assert spec.runner == "universal_dsl"
        assert spec.provenance.source.endswith(f"/{schema_name}.toml")
        assert spec.provenance.citation is not None
        assert spec.provenance.citation
        if "doi" in trace_name:
            assert spec.provenance.citation.startswith("doi:")


def test_lif_seed_features_match_independent_closed_form_solution() -> None:
    """Committed LIF features must match the closed-form RC solution, not the runner."""
    spec = load_reference_trace_spec("lif_constant_current_closed_form")

    expected = _closed_form_features(
        initial=-65.0,
        steady=-55.0,
        tau=10.0,
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_quadratic_if_trace_features_match_independent_analytic_solution() -> None:
    """Committed QIF features must match the analytic zero-current Riccati flow."""
    spec = load_reference_trace_spec("quadratic_if_zero_current_analytic")

    expected = _quadratic_if_zero_current_features(
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "quadratic_if"
    assert spec.provenance.citation == "doi:10.1152/jn.2000.83.2.808"
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_theta_trace_features_match_independent_phase_solution() -> None:
    """Committed theta features must match the tangent half-angle phase solution."""
    spec = load_reference_trace_spec("theta_constant_current_phase_analytic")

    expected = _theta_constant_current_features(
        current=spec.protocol.inputs["I"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "theta"
    assert spec.provenance.kind == "analytic_closed_form"
    assert spec.provenance.citation == "doi:10.1137/0146017"
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


_PARITY_CASES: list[tuple[str, str, str, str, Callable[[ReferenceTraceSpec], dict[str, float]]]] = [
    (
        "resonate_fire_subthreshold_resonance_doi",
        "resonate_fire",
        "analytic_linear_euler_reference",
        "doi:10.1162/089976601300014538",
        lambda spec: _resonate_fire_linear_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "glif_constant_current_threshold_adaptation",
        "glif",
        "independent_rk4_reference",
        "doi:10.1038/s41467-017-02717-4",
        lambda spec: _glif_driven_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "izhikevich_regular_spiking_doi",
        "izhikevich",
        "independent_euler_reference",
        "doi:10.1109/TNN.2003.820440",
        lambda spec: _izhikevich_rs_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "fitzhugh_nagumo_driven_oscillation_doi",
        "fitzhugh_nagumo",
        "independent_rk4_reference",
        "doi:10.1016/S0006-3495(61)86902-6",
        lambda spec: _fitzhugh_nagumo_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "fitzhugh_rinzel_driven_bursting_doi",
        "fitzhugh_rinzel",
        "independent_rk4_reference",
        "doi:10.1007/978-3-642-93360-8_26",
        lambda spec: _fitzhugh_rinzel_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "pernarowski_autonomous_bursting_doi",
        "pernarowski",
        "independent_rk4_reference",
        "doi:10.1137/S003613999223449X",
        lambda spec: _pernarowski_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "terman_wang_legion_oscillation_doi",
        "terman_wang",
        "independent_rk4_reference",
        "doi:10.1016/0167-2789(94)00205-5",
        lambda spec: _terman_wang_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "wilson_hr_driven_spiking_doi",
        "wilson_hr",
        "independent_rk4_reference",
        "doi:10.1006/jtbi.1999.1002",
        lambda spec: _wilson_hr_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "mckean_driven_oscillation_doi",
        "mckean",
        "independent_rk4_reference",
        "doi:10.1016/0001-8708(70)90023-X",
        lambda spec: _mckean_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "adex_resting_adaptation_doi",
        "adex",
        "independent_euler_reference",
        "doi:10.1152/jn.00686.2005",
        lambda spec: _adex_subthreshold_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "exp_if_resting_exponential_doi",
        "exp_if",
        "independent_euler_reference",
        "doi:10.1523/JNEUROSCI.23-37-11628.2003",
        lambda spec: _exp_if_subthreshold_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "hindmarsh_rose_short_bursting_prefix",
        "hindmarsh_rose",
        "independent_euler_reference",
        "doi:10.1098/rspb.1984.0024",
        lambda spec: _hindmarsh_rose_prefix_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "morris_lecar_driven_oscillation_doi",
        "morris_lecar",
        "independent_rk4_reference",
        "doi:10.1016/S0006-3495(81)84782-0",
        lambda spec: _morris_lecar_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "hodgkin_huxley_driven_spiking_doi",
        "hodgkin_huxley",
        "independent_macrostep_rk4_reference",
        "doi:10.1113/jphysiol.1952.sp004764",
        lambda spec: _hodgkin_huxley_macrostep_rk4_features(
            current=spec.protocol.inputs["I"],
            dt=spec.protocol.dt,
            steps=spec.protocol.steps,
            substeps=100,
        ),
    ),
    (
        "connor_stevens_driven_spiking_doi",
        "connor_stevens",
        "independent_macrostep_rk4_reference",
        "doi:10.1113/jphysiol.1971.sp009366",
        lambda spec: _connor_stevens_macrostep_rk4_features(
            current=spec.protocol.inputs["I"],
            dt=spec.protocol.dt,
            steps=spec.protocol.steps,
            substeps=100,
        ),
    ),
    (
        "wang_buzsaki_driven_spiking_doi",
        "wang_buzsaki",
        "independent_macrostep_gauss_seidel_reference",
        "doi:10.1523/JNEUROSCI.16-20-06402.1996",
        lambda spec: _wang_buzsaki_macrostep_gauss_seidel_features(
            current=spec.protocol.inputs["I"],
            dt=spec.protocol.dt,
            steps=spec.protocol.steps,
            substeps=50,
        ),
    ),
]


@pytest.mark.parametrize(
    ("trace_name", "schema_name", "kind", "citation", "reference"),
    _PARITY_CASES,
    ids=[case[1] for case in _PARITY_CASES],
)
def test_trace_features_match_independent_reference(
    trace_name: str,
    schema_name: str,
    kind: str,
    citation: str,
    reference: Callable[[ReferenceTraceSpec], dict[str, float]],
) -> None:
    """Each committed trace must reproduce an independent re-derivation to ``1e-12``.

    The per-case ``reference`` callable recomputes the expected feature map from the
    model's published equations (an explicit-Euler or analytic recurrence), so a
    passing assertion proves the committed corpus is independently reproduced rather
    than regenerated by the schema runner itself. The committed feature set must match
    the reference set exactly and every value to ``1e-12``.
    """
    spec = load_reference_trace_spec(trace_name)

    expected = reference(spec)

    assert spec.schema_name == schema_name
    assert spec.provenance.kind == kind
    assert spec.provenance.citation == citation
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_rulkov_map_trace_features_match_independent_map_iteration() -> None:
    """Committed Rulkov features must match an independent piecewise-map iteration."""
    spec = load_reference_trace_spec("rulkov_map_driven_spiking_doi")

    expected = _rulkov_map_features(
        current=spec.protocol.inputs["I"],
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "rulkov_map"
    assert spec.provenance.kind == "map_iteration_reference"
    assert spec.provenance.citation == "doi:10.1103/PhysRevE.65.041922"
    assert spec.expected_features["spike_count"] > 0
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


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
