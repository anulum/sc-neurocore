# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Formal SNN verification standard tests

from __future__ import annotations

import pytest

from sc_neurocore.verification import (
    SNNVerificationEvidence,
    SNNVerificationRequirement,
    SNNVerificationStandardProfile,
    VerificationClaimStatus,
    VerificationEvidenceKind,
    VerificationLevel,
    assess_snn_verification_standard,
    publication_grade_snn_standard_profile,
)


def _evidence(
    evidence_id: str,
    level: VerificationLevel,
    kind: VerificationEvidenceKind,
    status: VerificationClaimStatus = VerificationClaimStatus.PASS,
) -> SNNVerificationEvidence:
    return SNNVerificationEvidence(
        evidence_id=evidence_id,
        level=level,
        kind=kind,
        status=status,
        description=f"{evidence_id} evidence",
        artefact=f"{evidence_id}.json",
        digest="0" * 64,
    )


def test_publication_grade_profile_has_required_boundaries() -> None:
    profile = publication_grade_snn_standard_profile()
    requirement_ids = {item.requirement_id for item in profile.requirements}

    assert profile.profile_id == "publication-grade-snn-v1"
    assert "liveness_reachable_transition_fireability" in requirement_ids
    assert "boundedness_token_bounds" in requirement_ids
    assert "deadlock_freedom_compiler_output" in requirement_ids
    assert "implementation_equivalence" in requirement_ids
    assert "external_formal_proof" in requirement_ids
    assert any(not item.mandatory for item in profile.requirements)


def test_standard_passes_when_all_mandatory_evidence_passes() -> None:
    report = assess_snn_verification_standard(
        (
            _evidence(
                "temporal",
                VerificationLevel.TEMPORAL_PROPERTIES,
                VerificationEvidenceKind.TEMPORAL_RESULT,
            ),
            _evidence(
                "interval",
                VerificationLevel.INTERVAL_PROOF,
                VerificationEvidenceKind.INTERVAL_BOUND,
            ),
            _evidence(
                "equiv",
                VerificationLevel.IMPLEMENTATION_EQUIVALENCE,
                VerificationEvidenceKind.EQUIVALENCE_TEST,
            ),
            _evidence(
                "lean",
                VerificationLevel.EXTERNAL_FORMAL_PROOF,
                VerificationEvidenceKind.FORMAL_TOOL_LOG,
            ),
        )
    )

    assert report.passed
    assert report.mandatory_coverage == 1.0
    assert report.missing_mandatory == ()
    assert report.failed_mandatory == ()
    assert report.to_dict()["schema_version"].endswith(".v1")


def test_standard_fails_closed_when_external_proof_is_missing() -> None:
    report = assess_snn_verification_standard(
        (
            _evidence(
                "temporal",
                VerificationLevel.TEMPORAL_PROPERTIES,
                VerificationEvidenceKind.TEMPORAL_RESULT,
            ),
            _evidence(
                "interval",
                VerificationLevel.INTERVAL_PROOF,
                VerificationEvidenceKind.INTERVAL_BOUND,
            ),
            _evidence(
                "equiv",
                VerificationLevel.IMPLEMENTATION_EQUIVALENCE,
                VerificationEvidenceKind.EQUIVALENCE_TEST,
            ),
        )
    )

    assert not report.passed
    assert "external_formal_proof" in report.missing_mandatory
    assert report.mandatory_coverage == 0.8


def test_standard_fails_when_any_matching_mandatory_evidence_fails() -> None:
    report = assess_snn_verification_standard(
        (
            _evidence(
                "temporal",
                VerificationLevel.TEMPORAL_PROPERTIES,
                VerificationEvidenceKind.TEMPORAL_RESULT,
            ),
            _evidence(
                "interval",
                VerificationLevel.INTERVAL_PROOF,
                VerificationEvidenceKind.INTERVAL_BOUND,
            ),
            _evidence(
                "equiv",
                VerificationLevel.IMPLEMENTATION_EQUIVALENCE,
                VerificationEvidenceKind.EQUIVALENCE_TEST,
                VerificationClaimStatus.FAIL,
            ),
            _evidence(
                "lean",
                VerificationLevel.EXTERNAL_FORMAL_PROOF,
                VerificationEvidenceKind.FORMAL_TOOL_LOG,
            ),
        )
    )

    assert not report.passed
    assert "implementation_equivalence" in report.failed_mandatory


def test_wrong_evidence_kind_does_not_satisfy_requirement() -> None:
    report = assess_snn_verification_standard(
        (
            _evidence(
                "wrong",
                VerificationLevel.EXTERNAL_FORMAL_PROOF,
                VerificationEvidenceKind.TRACE,
            ),
        )
    )

    assert not report.passed
    assert "external_formal_proof" in report.missing_mandatory


def test_evidence_requires_id_and_description() -> None:
    with pytest.raises(ValueError, match="evidence_id"):
        SNNVerificationEvidence(
            evidence_id="",
            level=VerificationLevel.INTERVAL_PROOF,
            kind=VerificationEvidenceKind.INTERVAL_BOUND,
            status=VerificationClaimStatus.PASS,
            description="bad",
        )


def test_evidence_requires_non_empty_description() -> None:
    with pytest.raises(ValueError, match="description must be non-empty"):
        SNNVerificationEvidence(
            evidence_id="e1",
            level=VerificationLevel.INTERVAL_PROOF,
            kind=VerificationEvidenceKind.INTERVAL_BOUND,
            status=VerificationClaimStatus.PASS,
            description="",
        )


def _requirement(**overrides: object) -> SNNVerificationRequirement:
    kwargs: dict[str, object] = dict(
        requirement_id="r1",
        level=VerificationLevel.BOUNDED_SIMULATION,
        accepted_kinds=(VerificationEvidenceKind.TRACE,),
        description="a requirement",
    )
    kwargs.update(overrides)
    return SNNVerificationRequirement(**kwargs)  # type: ignore[arg-type]


def test_requirement_rejects_malformed_contracts() -> None:
    with pytest.raises(ValueError, match="requirement_id must be non-empty"):
        _requirement(requirement_id="")
    with pytest.raises(ValueError, match="accepted_kinds must not be empty"):
        _requirement(accepted_kinds=())
    with pytest.raises(ValueError, match="description must be non-empty"):
        _requirement(description="")


def test_profile_rejects_malformed_contracts() -> None:
    with pytest.raises(ValueError, match="profile_id must be non-empty"):
        SNNVerificationStandardProfile(
            profile_id="", description="d", requirements=(_requirement(),)
        )
    with pytest.raises(ValueError, match="description must be non-empty"):
        SNNVerificationStandardProfile(
            profile_id="p", description="", requirements=(_requirement(),)
        )
    with pytest.raises(ValueError, match="requirement ids must be unique"):
        SNNVerificationStandardProfile(
            profile_id="p",
            description="d",
            requirements=(_requirement(), _requirement()),
        )


def test_optional_requirement_with_only_missing_evidence_is_missing_and_fully_covered() -> None:
    # An optional requirement matched only by MISSING-status evidence resolves to
    # MISSING; with no mandatory requirements the mandatory coverage is a full 1.0.
    profile = SNNVerificationStandardProfile(
        profile_id="optional-only",
        description="profile with a single optional requirement",
        requirements=(_requirement(mandatory=False),),
    )
    evidence = _evidence(
        "e1",
        VerificationLevel.BOUNDED_SIMULATION,
        VerificationEvidenceKind.TRACE,
        status=VerificationClaimStatus.MISSING,
    )

    report = assess_snn_verification_standard([evidence], profile)

    assert report.requirement_results[0].status == VerificationClaimStatus.MISSING
    assert report.mandatory_coverage == 1.0
