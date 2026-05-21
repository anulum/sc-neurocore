# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Industrial application profile tests

from __future__ import annotations

import pytest

from sc_neurocore.industrial_applications import (
    EvidenceCategory,
    IndustrialApplicationRegistry,
    IndustrialDomain,
    assess_industrial_readiness,
    default_industrial_profiles,
)
from sc_neurocore.safety_cert import EvidenceBag, EvidenceItem, SafetyStandard


def _bag(*categories: str) -> EvidenceBag:
    evidence = EvidenceBag()
    for index, category in enumerate(categories):
        evidence.add(
            EvidenceItem(
                filename=f"evidence_{index}.dat",
                category=category,
                description=f"{category} evidence",
            )
        )
    return evidence


def test_default_profiles_cover_expected_domains_and_standards() -> None:
    profiles = default_industrial_profiles()
    domains = {profile.domain for profile in profiles}

    assert domains == set(IndustrialDomain)
    assert any(SafetyStandard.DO_254 in profile.safety_standards for profile in profiles)
    assert any(SafetyStandard.FDA_CLASS_III in profile.safety_standards for profile in profiles)
    assert all(profile.hazards for profile in profiles)
    assert all(profile.required_modules for profile in profiles)


def test_readiness_fails_closed_when_mandatory_evidence_is_missing() -> None:
    assessment = assess_industrial_readiness(
        IndustrialDomain.AEROSPACE,
        _bag("design", "formal"),
    )

    assert not assessment.ready
    assert assessment.mandatory_coverage < 1.0
    missing = {item.category for item in assessment.missing_mandatory}
    assert EvidenceCategory.TEST in missing
    assert EvidenceCategory.HIL in missing


def test_readiness_passes_when_required_categories_are_present() -> None:
    assessment = assess_industrial_readiness(
        "industrial_control",
        _bag("design", "test", "analysis", "report"),
    )

    assert assessment.ready
    assert assessment.mandatory_coverage == 1.0
    assert assessment.missing_mandatory == ()
    assert assessment.to_dict()["ready"] is True


def test_evidence_category_aliases_are_normalised() -> None:
    assessment = assess_industrial_readiness(
        IndustrialDomain.MEDICAL,
        _bag("design", "test", "analysis", "hardware-in-loop", "report", "security"),
    )

    assert assessment.ready
    assert EvidenceCategory.HIL in assessment.present_categories


def test_registry_rejects_unknown_domain() -> None:
    registry = IndustrialApplicationRegistry()

    with pytest.raises(ValueError, match="unknown industrial domain"):
        registry.get("unknown")


def test_profiles_are_returned_in_deterministic_order() -> None:
    registry = IndustrialApplicationRegistry()
    names = [profile.domain.value for profile in registry.list_profiles()]

    assert names == sorted(names)


def test_robotics_profile_requires_timing_evidence() -> None:
    assessment = assess_industrial_readiness(
        IndustrialDomain.ROBOTICS,
        _bag("design", "test", "analysis", "report"),
    )

    assert not assessment.ready
    assert EvidenceCategory.TIMING in {item.category for item in assessment.missing_mandatory}


def test_smart_grid_profile_accepts_latency_alias_for_timing() -> None:
    assessment = assess_industrial_readiness(
        IndustrialDomain.SMART_GRID,
        _bag("design", "latency", "test", "analysis", "report"),
    )

    assert assessment.ready
    assert EvidenceCategory.TIMING in assessment.present_categories


def test_fusion_control_profile_requires_hil_and_timing() -> None:
    assessment = assess_industrial_readiness(
        IndustrialDomain.FUSION_CONTROL,
        _bag("design", "timing", "test", "analysis", "report"),
    )

    assert not assessment.ready
    assert EvidenceCategory.HIL in {item.category for item in assessment.missing_mandatory}
