# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (enrolment) from former test_readiness_evidence_index.py

from __future__ import annotations

from readiness_evidence_index_support import *  # noqa: F403

def test_enrolled_shortlist_excludes_peer_wang_buzsaki_from_apply(tool: ModuleType) -> None:
    """Wang-Buzsaki remains inventored but skip_apply for peer-lane isolation."""
    wb = [e for e in tool.ENROLLED if e.schema_name == "wang_buzsaki"]
    assert len(wb) == 1
    assert wb[0].skip_apply is True
    assert "peer" in wb[0].skip_reason.lower() or "Gauss" in wb[0].skip_reason

def test_enrolled_class_names_exist_in_descriptor_corpus(tool: ModuleType) -> None:
    """Every enrolled class_name has an on-disk descriptor (except none expected)."""
    from sc_neurocore.neurons.model_catalogue import descriptor_path

    missing = [e.class_name for e in tool.ENROLLED if not descriptor_path(e.class_name).is_file()]
    assert missing == [], f"missing descriptors: {missing}"

def test_expif_is_enrolled_at_q3232_cosim_tier(tool: ModuleType) -> None:
    """ExpIF no longer inherits the obsolete compile-only schema-gap claim."""
    expif = next(e for e in tool.ENROLLED if e.class_name == "ExpIFNeuron")
    assert expif.level == "h1_cosim"
    assert expif.evidence == "tests/test_cosim_exp_if.py::test_expif_q3232_spike_parity"
    assert "Q32.32" in expif.operating_point
    assert "0/0/1/2/5/9" in expif.tolerance

def test_escape_rate_is_enrolled_with_class_correct_statistical_metric(
    tool: ModuleType,
) -> None:
    """Stochastic fidelity must never be relabelled as deterministic parity."""
    escape = next(e for e in tool.ENROLLED if e.class_name == "EscapeRateNeuron")
    assert escape.level == "h1_cosim"
    assert escape.metric == "statistical"
    assert escape.evidence == (
        "tests/test_cosim_escape_rate.py::"
        "test_seeded_full_period_event_stream_and_distribution_match_python"
    )
    assert "65,535-state" in escape.operating_point
    assert "ISI mean/CV" in escape.tolerance
    validation = tool.validation_section(escape, has_dynamics=True)
    assert validation["metric"] == "statistical"

def test_poisson_is_enrolled_with_full_period_statistical_cosim(tool: ModuleType) -> None:
    """Bind the stochastic event source to its registered and folded RTL evidence."""
    poisson = next(e for e in tool.ENROLLED if e.class_name == "PoissonNeuron")
    assert poisson.level == "h1_cosim"
    assert poisson.metric == "statistical"
    assert poisson.evidence == (
        "tests/test_cosim_poisson.py::"
        "test_seeded_full_period_registered_and_folded_streams_match_python"
    )
    assert "65,535-state" in poisson.operating_point
    assert "registered Q24.24 RTL" in poisson.tolerance
    validation = tool.validation_section(poisson, has_dynamics=True)
    assert validation["metric"] == "statistical"

def test_lapicque_is_enrolled_with_dedicated_exact_flow_evidence(tool: ModuleType) -> None:
    """Replace the generic suite pointer with the measured Lapicque contract."""
    lapicque = next(e for e in tool.ENROLLED if e.class_name == "LapicqueNeuron")
    assert lapicque.level == "h1_cosim"
    assert lapicque.evidence == (
        "tests/test_cosim_lapicque.py::test_q1616_preserves_event_vectors_and_voltage_bound"
    )
    assert "I=0.333,2.3,20.25" in lapicque.operating_point
    assert "Q16.16 event vectors exact" in lapicque.tolerance

def test_quadratic_if_is_enrolled_with_dedicated_exact_flow_evidence(
    tool: ModuleType,
) -> None:
    """Replace the removed generic-suite pointer with measured QIF evidence."""
    quadratic_if = next(e for e in tool.ENROLLED if e.class_name == "QuadraticIFNeuron")
    assert quadratic_if.level == "h1_cosim"
    assert quadratic_if.evidence == (
        "tests/test_cosim_quadratic_if.py::test_q1616_preserves_event_vectors_and_voltage_bound"
    )
    assert "I=0,0.333,0.5,1,2,5,20,50" in quadratic_if.operating_point
    assert "Q16.16 event vectors exact" in quadratic_if.tolerance

def test_theta_is_enrolled_with_dedicated_exact_flow_evidence(
    tool: ModuleType,
) -> None:
    """Replace the generic transcendental-suite pointer with measured Theta evidence."""
    theta = next(e for e in tool.ENROLLED if e.class_name == "ThetaNeuron")
    assert theta.level == "h1_cosim"
    assert theta.evidence == (
        "tests/test_cosim_theta.py::test_q1616_preserves_complete_event_count_vector"
    )
    assert "I=-1,-0.5,0,0.1,0.333,0.5,1,2,5,20,50" in theta.operating_point
    assert "Q16.16 event counts exact" in theta.tolerance
    assert "below 0.17 rad" in theta.tolerance
