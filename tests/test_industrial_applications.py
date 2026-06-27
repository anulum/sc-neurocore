# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Industrial application profile tests

"""Readiness-gate tests for industrial application profiles."""

from __future__ import annotations

import pytest

from sc_neurocore.industrial_applications import (
    EvidenceCategory,
    EvidenceRequirement,
    IndustrialApplicationProfile,
    IndustrialApplicationRegistry,
    IndustrialDomain,
    assess_industrial_readiness,
    default_industrial_profiles,
)
from sc_neurocore.safety_cert import EvidenceBag, EvidenceItem, SafetyStandard


def _bag(*categories: str) -> EvidenceBag:
    """Build an evidence bag with one synthetic item per category."""
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
    """Default profiles cover every declared domain and key standards."""
    profiles = default_industrial_profiles()
    domains = {profile.domain for profile in profiles}

    assert domains == set(IndustrialDomain)
    assert any(SafetyStandard.DO_254 in profile.safety_standards for profile in profiles)
    assert any(SafetyStandard.FDA_CLASS_III in profile.safety_standards for profile in profiles)
    assert all(profile.hazards for profile in profiles)
    assert all(profile.required_modules for profile in profiles)


def test_readiness_fails_closed_when_mandatory_evidence_is_missing() -> None:
    """Missing mandatory evidence keeps aerospace readiness closed."""
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
    """Industrial-control readiness passes when mandatory categories exist."""
    assessment = assess_industrial_readiness(
        "industrial_control",
        _bag("design", "test", "analysis", "report"),
    )

    assert assessment.ready
    assert assessment.mandatory_coverage == 1.0
    assert assessment.missing_mandatory == ()
    assert assessment.to_dict()["ready"] is True


def test_evidence_category_aliases_are_normalised() -> None:
    """Evidence aliases normalise into canonical readiness categories."""
    assessment = assess_industrial_readiness(
        IndustrialDomain.MEDICAL,
        _bag("design", "test", "analysis", "hardware-in-loop", "report", "security"),
    )

    assert assessment.ready
    assert EvidenceCategory.HIL in assessment.present_categories


def test_registry_rejects_unknown_domain() -> None:
    """The registry rejects domain names outside the declared enum."""
    registry = IndustrialApplicationRegistry()

    with pytest.raises(ValueError, match="unknown industrial domain"):
        registry.get("unknown")


def test_profiles_are_returned_in_deterministic_order() -> None:
    """Registry listing order is deterministic for docs and callers."""
    registry = IndustrialApplicationRegistry()
    names = [profile.domain.value for profile in registry.list_profiles()]

    assert names == sorted(names)


def test_robotics_profile_requires_timing_evidence() -> None:
    """Robotics readiness requires mandatory timing evidence."""
    assessment = assess_industrial_readiness(
        IndustrialDomain.ROBOTICS,
        _bag("design", "test", "analysis", "report"),
    )

    assert not assessment.ready
    assert EvidenceCategory.TIMING in {item.category for item in assessment.missing_mandatory}


def test_smart_grid_profile_accepts_latency_alias_for_timing() -> None:
    """Smart-grid readiness accepts latency evidence as timing evidence."""
    assessment = assess_industrial_readiness(
        IndustrialDomain.SMART_GRID,
        _bag("design", "latency", "test", "analysis", "report"),
    )

    assert assessment.ready
    assert EvidenceCategory.TIMING in assessment.present_categories


def test_fusion_control_profile_requires_hil_and_timing() -> None:
    """Fusion-control readiness fails when HIL evidence is absent."""
    assessment = assess_industrial_readiness(
        IndustrialDomain.FUSION_CONTROL,
        _bag("design", "timing", "test", "analysis", "report"),
    )

    assert not assessment.ready
    assert EvidenceCategory.HIL in {item.category for item in assessment.missing_mandatory}


def _optional_only_profile(domain: IndustrialDomain) -> IndustrialApplicationProfile:
    """Create a profile whose evidence requirements are all optional."""
    return IndustrialApplicationProfile(
        domain=domain,
        name="optional-only profile",
        description="profile whose evidence requirements are all optional",
        safety_standards=(SafetyStandard.IEC_61508,),
        target_sil=None,
        target_asil=None,
        hazards=("illustrative hazard",),
        required_modules=("safety_cert",),
        evidence_requirements=(
            EvidenceRequirement(
                category=EvidenceCategory.DESIGN,
                description="optional design evidence",
                mandatory=False,
            ),
        ),
    )


def test_all_optional_profile_is_fully_covered_yet_lists_optional_gaps() -> None:
    """Profiles with no mandatory evidence report full mandatory coverage."""
    registry = IndustrialApplicationRegistry(
        profiles=[_optional_only_profile(IndustrialDomain.AEROSPACE)]
    )

    assessment = registry.assess(IndustrialDomain.AEROSPACE, EvidenceBag())

    # No evidence supplied, so the optional requirement is recorded as a gap...
    assert len(assessment.missing_optional) == 1
    assert assessment.missing_mandatory == ()
    # ...but with no mandatory requirements the mandatory coverage is a full 1.0.
    assert assessment.mandatory_coverage == 1.0


def test_registry_rejects_valid_domain_without_a_registered_profile() -> None:
    """A valid enum domain is rejected when no profile is registered."""
    registry = IndustrialApplicationRegistry(
        profiles=[_optional_only_profile(IndustrialDomain.AEROSPACE)]
    )

    # AUTOMOTIVE is a valid domain enum but is absent from this single-profile registry.
    with pytest.raises(ValueError, match="unknown industrial domain"):
        registry.get(IndustrialDomain.AUTOMOTIVE)
