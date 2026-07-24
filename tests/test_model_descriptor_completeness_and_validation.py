# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (completeness_and_validation) from former test_model_descriptor.py

from __future__ import annotations

from tests.model_descriptor_support import *  # noqa: F403

def test_completeness_tier_zero_for_descriptor_without_structure() -> None:
    """Descriptors with no parameters and no state remain at tier zero."""

    payload = _minimal_payload()
    payload["state"] = {}
    payload["parameters"] = {}

    descriptor = parse_model_descriptor(payload)

    assert descriptor.parameters == ()
    assert descriptor.state == ()
    assert descriptor_completeness_tier(descriptor) == 0


def test_completeness_tiers_rise_with_curation() -> None:
    """Each curation column lifts the descriptor to the next tier."""

    payload = _minimal_payload()
    payload["metadata"].update({"family": "Integrate-and-Fire", "category": "adaptive"})
    assert descriptor_completeness_tier(parse_model_descriptor(payload)) == 1

    # Tier 2 — scientifically curated: citeable provenance + every parameter curated.
    payload["provenance"] = {"authors": ["Brette"], "year": 2005, "doi": "10.1152/jn.00686.2005"}
    payload["parameters"] = {
        "tau": {"default": 20.0, "unit": "ms", "range": [1.0, 100.0], "meaning": "time constant"}
    }
    assert descriptor_completeness_tier(parse_model_descriptor(payload)) == 2

    # Tier 3 — engineering-verified: two implemented backends + a golden trace.
    payload["backends"] = {
        "python": {"status": "implemented"},
        "rust": {"status": "implemented", "parity": "ulp-bounded"},
    }
    payload["reproducibility"] = {
        "reference_config": "golden/adex.json",
        "golden_trace_sha256": "a" * 64,
    }
    assert descriptor_completeness_tier(parse_model_descriptor(payload)) == 3


def test_validation_defaults_are_empty_and_unvalidated() -> None:
    """An absent [validation] section yields an empty, unvalidated facet."""
    descriptor = parse_model_descriptor(_minimal_payload())
    assert descriptor.validation == Validation()
    assert descriptor.validation.metric == "none"
    assert descriptor.validation.dynamics_faithful is False
    assert descriptor.validation.is_class_validated is False


def test_validation_is_class_validated_needs_metric_and_evidence() -> None:
    """The validated predicate requires both a non-trivial metric and evidence."""
    assert Validation(metric="parity", evidence="trace.json").is_class_validated is True
    assert Validation(metric="parity").is_class_validated is False
    assert Validation(metric="none", evidence="trace.json").is_class_validated is False


def test_parse_validation_section_reads_every_field() -> None:
    """The [validation] section round-trips its recorded evidence fields."""
    payload = _minimal_payload()
    payload["validation"] = {
        "dynamics_faithful": True,
        "metric": "statistical",
        "operating_point": "Poisson drive 20 Hz",
        "tolerance": "KS < 0.05",
        "evidence": "golden/adex_stats.json",
    }
    validation = parse_model_descriptor(payload).validation
    assert validation.dynamics_faithful is True
    assert validation.metric == "statistical"
    assert validation.operating_point == "Poisson drive 20 Hz"
    assert validation.tolerance == "KS < 0.05"
    assert validation.evidence == "golden/adex_stats.json"
    assert validation.is_class_validated is True


def test_parse_rejects_unknown_validation_metric() -> None:
    payload = _minimal_payload()
    payload["validation"] = {"metric": "vibes"}
    with pytest.raises(ModelDescriptorError, match="validation metric"):
        parse_model_descriptor(payload)


def test_parse_rejects_non_boolean_evidence_flag() -> None:
    payload = _minimal_payload()
    payload["validation"] = {"dynamics_faithful": "yes"}
    with pytest.raises(ModelDescriptorError, match="dynamics_faithful"):
        parse_model_descriptor(payload)


