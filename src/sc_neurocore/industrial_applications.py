# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Industrial application readiness profiles

"""Industrial application profiles and evidence-readiness assessment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from sc_neurocore.safety_cert import ASILLevel, EvidenceBag, SafetyStandard, SILLevel


class IndustrialDomain(Enum):
    """Supported industrial application domains."""

    AEROSPACE = "aerospace"
    AUTOMOTIVE = "automotive"
    MEDICAL = "medical"
    RAIL = "rail"
    INDUSTRIAL_CONTROL = "industrial_control"


class EvidenceCategory(Enum):
    """Evidence categories expected in an industrial readiness pack."""

    DESIGN = "design"
    FORMAL = "formal"
    TEST = "test"
    ANALYSIS = "analysis"
    REPORT = "report"
    HIL = "hil"
    SECURITY = "security"


@dataclass(frozen=True)
class EvidenceRequirement:
    """One evidence requirement for an application profile."""

    category: EvidenceCategory
    description: str
    mandatory: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready requirement."""
        return {
            "category": self.category.value,
            "description": self.description,
            "mandatory": self.mandatory,
        }


@dataclass(frozen=True)
class IndustrialApplicationProfile:
    """Readiness profile for one SC-NeuroCore industrial use case."""

    domain: IndustrialDomain
    name: str
    description: str
    safety_standards: tuple[SafetyStandard, ...]
    target_sil: SILLevel | None
    target_asil: ASILLevel | None
    hazards: tuple[str, ...]
    required_modules: tuple[str, ...]
    evidence_requirements: tuple[EvidenceRequirement, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready profile."""
        return {
            "domain": self.domain.value,
            "name": self.name,
            "description": self.description,
            "safety_standards": [standard.value for standard in self.safety_standards],
            "target_sil": self.target_sil.name if self.target_sil is not None else None,
            "target_asil": self.target_asil.value if self.target_asil is not None else None,
            "hazards": list(self.hazards),
            "required_modules": list(self.required_modules),
            "evidence_requirements": [
                requirement.to_dict() for requirement in self.evidence_requirements
            ],
        }


@dataclass(frozen=True)
class IndustrialReadinessAssessment:
    """Evidence coverage assessment for one industrial application profile."""

    profile: IndustrialApplicationProfile
    present_categories: tuple[EvidenceCategory, ...]
    missing_mandatory: tuple[EvidenceRequirement, ...]
    missing_optional: tuple[EvidenceRequirement, ...]

    @property
    def ready(self) -> bool:
        """Whether all mandatory evidence categories are present."""
        return not self.missing_mandatory

    @property
    def mandatory_coverage(self) -> float:
        """Mandatory evidence coverage ratio."""
        mandatory = [item for item in self.profile.evidence_requirements if item.mandatory]
        if not mandatory:
            return 1.0
        covered = len(mandatory) - len(self.missing_mandatory)
        return covered / len(mandatory)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready assessment."""
        return {
            "profile": self.profile.to_dict(),
            "ready": self.ready,
            "mandatory_coverage": self.mandatory_coverage,
            "present_categories": [category.value for category in self.present_categories],
            "missing_mandatory": [item.to_dict() for item in self.missing_mandatory],
            "missing_optional": [item.to_dict() for item in self.missing_optional],
        }


class IndustrialApplicationRegistry:
    """Registry of application profiles and evidence-readiness checks."""

    def __init__(
        self,
        profiles: Iterable[IndustrialApplicationProfile] | None = None,
    ) -> None:
        self._profiles = {
            profile.domain: profile for profile in (profiles or default_industrial_profiles())
        }

    def get(self, domain: IndustrialDomain | str) -> IndustrialApplicationProfile:
        """Return the profile for a domain."""
        try:
            key = IndustrialDomain(domain) if isinstance(domain, str) else domain
        except ValueError as exc:
            raise ValueError(f"unknown industrial domain {domain!r}") from exc
        try:
            return self._profiles[key]
        except KeyError as exc:
            raise ValueError(f"unknown industrial domain {key.value!r}") from exc

    def list_profiles(self) -> tuple[IndustrialApplicationProfile, ...]:
        """Return all registered profiles in deterministic order."""
        return tuple(self._profiles[key] for key in sorted(self._profiles, key=lambda item: item.value))

    def assess(
        self,
        domain: IndustrialDomain | str,
        evidence_bag: EvidenceBag,
    ) -> IndustrialReadinessAssessment:
        """Assess whether the evidence bag covers a domain profile."""
        profile = self.get(domain)
        present = _normalise_evidence_categories(item.category for item in evidence_bag.items)
        missing_mandatory = []
        missing_optional = []
        for requirement in profile.evidence_requirements:
            if requirement.category in present:
                continue
            if requirement.mandatory:
                missing_mandatory.append(requirement)
            else:
                missing_optional.append(requirement)
        return IndustrialReadinessAssessment(
            profile=profile,
            present_categories=tuple(sorted(present, key=lambda item: item.value)),
            missing_mandatory=tuple(missing_mandatory),
            missing_optional=tuple(missing_optional),
        )


def default_industrial_profiles() -> tuple[IndustrialApplicationProfile, ...]:
    """Return conservative built-in industrial application profiles."""
    return (
        IndustrialApplicationProfile(
            domain=IndustrialDomain.AEROSPACE,
            name="Radiation-aware event-stream inference payload",
            description="SC bitstream inference for airborne or spaceborne event-stream workloads.",
            safety_standards=(SafetyStandard.DO_254, SafetyStandard.IEC_61508),
            target_sil=SILLevel.SIL_3,
            target_asil=None,
            hazards=(
                "single-event upset corrupts bitstream state",
                "timing violation masks watchdog response",
                "unqualified model update changes deployed behaviour",
            ),
            required_modules=(
                "fault_injection",
                "safety_cert",
                "hdl_gen.safety",
                "nir_bridge",
            ),
            evidence_requirements=_requirements(
                EvidenceCategory.DESIGN,
                EvidenceCategory.FORMAL,
                EvidenceCategory.TEST,
                EvidenceCategory.ANALYSIS,
                EvidenceCategory.HIL,
                EvidenceCategory.REPORT,
            ),
        ),
        IndustrialApplicationProfile(
            domain=IndustrialDomain.AUTOMOTIVE,
            name="ASIL-oriented edge perception accelerator",
            description="Deterministic SC inference path for bounded automotive edge perception.",
            safety_standards=(SafetyStandard.ISO_26262, SafetyStandard.IEC_61508),
            target_sil=SILLevel.SIL_2,
            target_asil=ASILLevel.ASIL_B,
            hazards=(
                "stale sensor frame used after deadline",
                "silent weight corruption changes decision boundary",
                "fallback mode unavailable after accelerator fault",
            ),
            required_modules=("safety_cert", "safety_monitor", "fault_injection", "nir_bridge"),
            evidence_requirements=_requirements(
                EvidenceCategory.DESIGN,
                EvidenceCategory.TEST,
                EvidenceCategory.ANALYSIS,
                EvidenceCategory.REPORT,
                EvidenceCategory.SECURITY,
            ),
        ),
        IndustrialApplicationProfile(
            domain=IndustrialDomain.MEDICAL,
            name="Research HIL neuromorphic signal-processing path",
            description="Closed-loop HIL research pipeline with explicit non-clinical boundary.",
            safety_standards=(SafetyStandard.FDA_CLASS_III, SafetyStandard.IEC_61508),
            target_sil=SILLevel.SIL_3,
            target_asil=None,
            hazards=(
                "feedback command exceeds bounded amplitude",
                "latency budget violation invalidates HIL assumption",
                "clinical use attempted without external safety case",
            ),
            required_modules=("bci_studio", "interfaces.bci_closed_loop", "safety_cert"),
            evidence_requirements=_requirements(
                EvidenceCategory.DESIGN,
                EvidenceCategory.TEST,
                EvidenceCategory.ANALYSIS,
                EvidenceCategory.HIL,
                EvidenceCategory.REPORT,
                EvidenceCategory.SECURITY,
            ),
        ),
        IndustrialApplicationProfile(
            domain=IndustrialDomain.RAIL,
            name="Deterministic safety-monitoring coprocessor",
            description="SC safety-monitoring and anomaly-detection coprocessor for rail systems.",
            safety_standards=(SafetyStandard.EN_50129, SafetyStandard.IEC_61508),
            target_sil=SILLevel.SIL_4,
            target_asil=None,
            hazards=(
                "dangerous undetected diagnostic failure",
                "traceability gap between requirement and monitor property",
                "field update bypasses proof evidence",
            ),
            required_modules=("safety_cert", "verification", "hdl_gen.safety"),
            evidence_requirements=_requirements(
                EvidenceCategory.DESIGN,
                EvidenceCategory.FORMAL,
                EvidenceCategory.TEST,
                EvidenceCategory.ANALYSIS,
                EvidenceCategory.REPORT,
                EvidenceCategory.SECURITY,
            ),
        ),
        IndustrialApplicationProfile(
            domain=IndustrialDomain.INDUSTRIAL_CONTROL,
            name="Condition-monitoring and predictive-maintenance edge node",
            description="SC anomaly detection for bounded industrial sensing workloads.",
            safety_standards=(SafetyStandard.IEC_61508,),
            target_sil=SILLevel.SIL_1,
            target_asil=None,
            hazards=(
                "missed anomaly due to unvalidated threshold drift",
                "correlated stochastic streams reduce diagnostic sensitivity",
                "incomplete calibration evidence hides sensor degradation",
            ),
            required_modules=("stochastic_doctor", "fault_injection", "safety_cert"),
            evidence_requirements=_requirements(
                EvidenceCategory.DESIGN,
                EvidenceCategory.TEST,
                EvidenceCategory.ANALYSIS,
                EvidenceCategory.REPORT,
            ),
        ),
    )


def assess_industrial_readiness(
    domain: IndustrialDomain | str,
    evidence_bag: EvidenceBag,
) -> IndustrialReadinessAssessment:
    """Assess readiness for a built-in industrial application domain."""
    return IndustrialApplicationRegistry().assess(domain, evidence_bag)


def _requirements(*categories: EvidenceCategory) -> tuple[EvidenceRequirement, ...]:
    return tuple(
        EvidenceRequirement(
            category=category,
            description=f"{category.value} evidence for domain-specific safety case",
        )
        for category in categories
    )


def _normalise_evidence_categories(categories: Iterable[str]) -> set[EvidenceCategory]:
    result: set[EvidenceCategory] = set()
    aliases = {
        "formal": EvidenceCategory.FORMAL,
        "test": EvidenceCategory.TEST,
        "analysis": EvidenceCategory.ANALYSIS,
        "design": EvidenceCategory.DESIGN,
        "report": EvidenceCategory.REPORT,
        "hil": EvidenceCategory.HIL,
        "hardware-in-loop": EvidenceCategory.HIL,
        "hardware_in_loop": EvidenceCategory.HIL,
        "security": EvidenceCategory.SECURITY,
    }
    for category in categories:
        key = category.strip().lower().replace(" ", "_")
        if key in aliases:
            result.add(aliases[key])
    return result
