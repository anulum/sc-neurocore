# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Formal SNN verification standard

"""Formal SNN verification standard profiles and conformance reports.

The standard is intentionally explicit about its boundary: it aggregates
bounded simulation, interval proof, temporal property, implementation, and
external formal-tool evidence into a machine-readable conformance report. It
does not claim unbounded semantic correctness unless an external proof artefact
is supplied and marked passing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


SCHEMA_VERSION = "sc-neurocore.formal-snn-verification-standard.v1"


class VerificationLevel(Enum):
    """Evidence levels for SNN verification claims."""

    BOUNDED_SIMULATION = "bounded_simulation"
    INTERVAL_PROOF = "interval_proof"
    TEMPORAL_PROPERTIES = "temporal_properties"
    IMPLEMENTATION_EQUIVALENCE = "implementation_equivalence"
    EXTERNAL_FORMAL_PROOF = "external_formal_proof"


class VerificationEvidenceKind(Enum):
    """Kinds of evidence accepted by the standard."""

    TEMPORAL_RESULT = "temporal_result"
    INTERVAL_BOUND = "interval_bound"
    FORMAL_TOOL_LOG = "formal_tool_log"
    HDL_ASSERTION = "hdl_assertion"
    EQUIVALENCE_TEST = "equivalence_test"
    TRACE = "trace"
    SAFETY_CASE = "safety_case"


class VerificationClaimStatus(Enum):
    """Status of one evidence item or standard requirement."""

    # Verification outcome label, not a credential.
    PASS = "pass"  # nosec B105
    FAIL = "fail"
    MISSING = "missing"


@dataclass(frozen=True)
class SNNVerificationEvidence:
    """One evidence item used in a formal SNN verification claim."""

    evidence_id: str
    kind: VerificationEvidenceKind
    level: VerificationLevel
    status: VerificationClaimStatus
    description: str
    artefact: str = ""
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("evidence_id must be non-empty")
        if not self.description:
            raise ValueError("description must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready evidence record."""
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind.value,
            "level": self.level.value,
            "status": self.status.value,
            "description": self.description,
            "artefact": self.artefact,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class SNNVerificationRequirement:
    """One mandatory or optional requirement in a standard profile."""

    requirement_id: str
    level: VerificationLevel
    accepted_kinds: tuple[VerificationEvidenceKind, ...]
    description: str
    mandatory: bool = True

    def __post_init__(self) -> None:
        if not self.requirement_id:
            raise ValueError("requirement_id must be non-empty")
        if not self.accepted_kinds:
            raise ValueError("accepted_kinds must not be empty")
        if not self.description:
            raise ValueError("description must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready requirement."""
        return {
            "requirement_id": self.requirement_id,
            "level": self.level.value,
            "accepted_kinds": [kind.value for kind in self.accepted_kinds],
            "description": self.description,
            "mandatory": self.mandatory,
        }


@dataclass(frozen=True)
class SNNVerificationStandardProfile:
    """Named set of requirements for a formal SNN verification claim."""

    profile_id: str
    description: str
    requirements: tuple[SNNVerificationRequirement, ...]

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        if not self.description:
            raise ValueError("description must be non-empty")
        ids = [requirement.requirement_id for requirement in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("requirement ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready profile."""
        return {
            "profile_id": self.profile_id,
            "description": self.description,
            "requirements": [requirement.to_dict() for requirement in self.requirements],
        }


@dataclass(frozen=True)
class SNNVerificationRequirementResult:
    """Evaluation of one profile requirement."""

    requirement: SNNVerificationRequirement
    status: VerificationClaimStatus
    matched_evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready requirement result."""
        return {
            "requirement": self.requirement.to_dict(),
            "status": self.status.value,
            "matched_evidence_ids": list(self.matched_evidence_ids),
        }


@dataclass(frozen=True)
class SNNVerificationConformanceReport:
    """Conformance report for a profile and evidence set."""

    schema_version: str
    profile: SNNVerificationStandardProfile
    requirement_results: tuple[SNNVerificationRequirementResult, ...]
    evidence: tuple[SNNVerificationEvidence, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """Whether all mandatory requirements passed."""
        return all(
            result.status == VerificationClaimStatus.PASS
            for result in self.requirement_results
            if result.requirement.mandatory
        )

    @property
    def missing_mandatory(self) -> tuple[str, ...]:
        """Mandatory requirement ids with missing evidence."""
        return tuple(
            result.requirement.requirement_id
            for result in self.requirement_results
            if result.requirement.mandatory and result.status == VerificationClaimStatus.MISSING
        )

    @property
    def failed_mandatory(self) -> tuple[str, ...]:
        """Mandatory requirement ids with failing evidence."""
        return tuple(
            result.requirement.requirement_id
            for result in self.requirement_results
            if result.requirement.mandatory and result.status == VerificationClaimStatus.FAIL
        )

    @property
    def mandatory_coverage(self) -> float:
        """Coverage ratio for mandatory requirements."""
        mandatory = [item for item in self.requirement_results if item.requirement.mandatory]
        if not mandatory:
            return 1.0
        covered = sum(item.status == VerificationClaimStatus.PASS for item in mandatory)
        return covered / len(mandatory)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready conformance report."""
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "mandatory_coverage": self.mandatory_coverage,
            "missing_mandatory": list(self.missing_mandatory),
            "failed_mandatory": list(self.failed_mandatory),
            "profile": self.profile.to_dict(),
            "requirement_results": [item.to_dict() for item in self.requirement_results],
            "evidence": [item.to_dict() for item in self.evidence],
        }


class SNNVerificationStandard:
    """Evaluate SNN verification evidence against a standard profile."""

    def __init__(self, profile: SNNVerificationStandardProfile | None = None) -> None:
        self.profile = profile or publication_grade_snn_standard_profile()

    def assess(
        self,
        evidence: Iterable[SNNVerificationEvidence],
    ) -> SNNVerificationConformanceReport:
        """Assess evidence against the configured profile."""
        evidence_items = tuple(evidence)
        results = tuple(
            self._assess_requirement(requirement, evidence_items)
            for requirement in self.profile.requirements
        )
        return SNNVerificationConformanceReport(
            schema_version=SCHEMA_VERSION,
            profile=self.profile,
            requirement_results=results,
            evidence=evidence_items,
        )

    @staticmethod
    def _assess_requirement(
        requirement: SNNVerificationRequirement,
        evidence_items: tuple[SNNVerificationEvidence, ...],
    ) -> SNNVerificationRequirementResult:
        matched = tuple(
            item
            for item in evidence_items
            if item.level == requirement.level and item.kind in requirement.accepted_kinds
        )
        if not matched:
            return SNNVerificationRequirementResult(
                requirement=requirement,
                status=VerificationClaimStatus.MISSING,
            )
        matched_ids = tuple(item.evidence_id for item in matched)
        if any(item.status == VerificationClaimStatus.FAIL for item in matched):
            return SNNVerificationRequirementResult(
                requirement=requirement,
                status=VerificationClaimStatus.FAIL,
                matched_evidence_ids=matched_ids,
            )
        if any(item.status == VerificationClaimStatus.PASS for item in matched):
            return SNNVerificationRequirementResult(
                requirement=requirement,
                status=VerificationClaimStatus.PASS,
                matched_evidence_ids=matched_ids,
            )
        return SNNVerificationRequirementResult(
            requirement=requirement,
            status=VerificationClaimStatus.MISSING,
            matched_evidence_ids=matched_ids,
        )


def publication_grade_snn_standard_profile() -> SNNVerificationStandardProfile:
    """Return the default formal SNN verification standard profile."""
    return SNNVerificationStandardProfile(
        profile_id="publication-grade-snn-v1",
        description=(
            "Minimum evidence profile for a scientifically defensible SNN verification claim: "
            "bounded temporal properties, interval bounds, implementation equivalence, and "
            "external formal proof evidence."
        ),
        requirements=(
            SNNVerificationRequirement(
                requirement_id="bounded_temporal_properties",
                level=VerificationLevel.TEMPORAL_PROPERTIES,
                accepted_kinds=(
                    VerificationEvidenceKind.TEMPORAL_RESULT,
                    VerificationEvidenceKind.TRACE,
                ),
                description="Temporal safety/liveness properties evaluated over declared bounds.",
            ),
            SNNVerificationRequirement(
                requirement_id="probability_interval_bounds",
                level=VerificationLevel.INTERVAL_PROOF,
                accepted_kinds=(VerificationEvidenceKind.INTERVAL_BOUND,),
                description="Interval arithmetic or equivalent proof of probability/state bounds.",
            ),
            SNNVerificationRequirement(
                requirement_id="implementation_equivalence",
                level=VerificationLevel.IMPLEMENTATION_EQUIVALENCE,
                accepted_kinds=(
                    VerificationEvidenceKind.EQUIVALENCE_TEST,
                    VerificationEvidenceKind.HDL_ASSERTION,
                ),
                description="Evidence that reference model and deployable implementation agree.",
            ),
            SNNVerificationRequirement(
                requirement_id="external_formal_proof",
                level=VerificationLevel.EXTERNAL_FORMAL_PROOF,
                accepted_kinds=(VerificationEvidenceKind.FORMAL_TOOL_LOG,),
                description="External prover/model-checker log for the stated formal boundary.",
            ),
            SNNVerificationRequirement(
                requirement_id="safety_case_traceability",
                level=VerificationLevel.BOUNDED_SIMULATION,
                accepted_kinds=(VerificationEvidenceKind.SAFETY_CASE,),
                description="Human-readable safety case tying assumptions, bounds, and artefacts.",
                mandatory=False,
            ),
        ),
    )


def assess_snn_verification_standard(
    evidence: Iterable[SNNVerificationEvidence],
    profile: SNNVerificationStandardProfile | None = None,
) -> SNNVerificationConformanceReport:
    """Assess evidence against the default or supplied SNN verification profile."""
    return SNNVerificationStandard(profile).assess(evidence)
