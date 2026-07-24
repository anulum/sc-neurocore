# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (science_axis) from former test_descriptor_tiers.py

from __future__ import annotations

from tests.descriptor_tiers_support import *  # noqa: F403

@pytest.mark.parametrize("expected", [0, 1, 2, 3])
def test_science_axis_matches_kernel_through_s3(expected: int) -> None:
    """S0-S3 track the curation kernel exactly, with no validation evidence."""
    payload = _s3_payload()
    payload.pop("validation", None)
    if expected < 3:
        payload["backends"] = {}  # drop the engineering-verified rung
        payload["reproducibility"] = {}
    if expected < 2:
        payload["provenance"] = {}  # drop citeable provenance
    if expected < 1:
        del payload["metadata"]["family"]
        del payload["metadata"]["category"]

    descriptor = parse_model_descriptor(payload)
    assert descriptor_completeness_tier(descriptor) == expected
    assert science_tier(descriptor) == expected


def test_science_tier_stays_at_base_when_below_three_despite_validation() -> None:
    """S4/S5 are gated behind the S3 kernel: validation alone cannot lift S0."""
    payload = _s0_payload()
    payload["dynamics"] = {"v": "-v"}
    payload["validation"] = {
        "dynamics_faithful": True,
        "metric": "parity",
        "evidence": "somewhere",
    }
    descriptor = parse_model_descriptor(payload)
    assert science_tier(descriptor) == 0


def test_science_tier_s4_requires_faithful_dynamics() -> None:
    """S4 opens only when dynamics are declared and confirmed faithful."""
    at_s3 = parse_model_descriptor(_s3_payload())
    assert science_tier(at_s3) == 3  # dynamics present but not confirmed faithful

    faithful = _s3_payload()
    faithful["validation"] = {"dynamics_faithful": True}
    assert science_tier(parse_model_descriptor(faithful)) == 4


def test_science_tier_s4_needs_declared_dynamics() -> None:
    """A faithful flag without any declared dynamics cannot reach S4."""
    payload = _s3_payload()
    payload["dynamics"] = {}
    payload["validation"] = {"dynamics_faithful": True}
    assert science_tier(parse_model_descriptor(payload)) == 3


def test_science_tier_s5_requires_metric_and_evidence() -> None:
    """S5 opens only with a non-trivial metric and committed evidence."""
    assert science_tier(parse_model_descriptor(_validated_payload())) == 5

    metric_only = _validated_payload()
    metric_only["validation"]["evidence"] = ""
    assert science_tier(parse_model_descriptor(metric_only)) == 4

    evidence_only = _validated_payload()
    evidence_only["validation"]["metric"] = "none"
    assert science_tier(parse_model_descriptor(evidence_only)) == 4
