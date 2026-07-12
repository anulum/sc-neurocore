# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Automated Safety & Regulatory Certification Generator

"""Requirement records and traceability-matrix reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

from sc_neurocore.safety_cert.standards import SILLevel, SafetyStandard


@dataclass
class Requirement:
    """One safety requirement with traceability links."""

    req_id: str
    description: str
    standard: SafetyStandard
    sil_level: SILLevel = SILLevel.SIL_2
    implementation_refs: List[str] = field(default_factory=list)
    verification_refs: List[str] = field(default_factory=list)
    status: str = "open"

    def __post_init__(self) -> None:
        """Validate the requirement and its initial traceability links."""
        if not isinstance(self.req_id, str) or not self.req_id.strip():
            raise ValueError("req_id must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.standard, SafetyStandard):
            raise ValueError("standard must be a SafetyStandard")
        if not isinstance(self.sil_level, SILLevel):
            raise ValueError("sil_level must be a SILLevel")
        if not isinstance(self.status, str) or self.status not in {
            "open",
            "implemented",
            "verified",
        }:
            raise ValueError("status must be one of: open, implemented, verified")

        for impl_ref in self.implementation_refs:
            if not isinstance(impl_ref, str) or not impl_ref.strip():
                raise ValueError("implementation_refs must contain non-empty strings")
        for verif_ref in self.verification_refs:
            if not isinstance(verif_ref, str) or not verif_ref.strip():
                raise ValueError("verification_refs must contain non-empty strings")


class TraceabilityMatrix:
    """Requirement → implementation → verification traceability.

    Each requirement links to:
    - Implementation artifacts (RTL files, Python modules)
    - Verification artifacts (formal proofs, UVM results, tests)
    """

    def __init__(self) -> None:
        self.requirements: Dict[str, Requirement] = {}

    def add_requirement(self, req: Requirement) -> None:
        """Add one uniquely identified requirement."""
        if not isinstance(req, Requirement):
            raise ValueError("req must be a Requirement")
        if req.req_id in self.requirements:
            raise ValueError(f"requirement already exists: {req.req_id}")
        self.requirements[req.req_id] = req

    def link_implementation(self, req_id: str, impl_ref: str) -> bool:
        """Link an implementation artifact, returning false for an unknown ID."""
        if not isinstance(req_id, str) or not req_id.strip():
            raise ValueError("req_id must be a non-empty string")
        if not isinstance(impl_ref, str) or not impl_ref.strip():
            raise ValueError("impl_ref must be a non-empty string")
        req_id = req_id.strip()
        impl_ref = impl_ref.strip()
        req = self.requirements.get(req_id)
        if req is None:
            return False
        if not isinstance(req, Requirement):
            raise ValueError("requirements must contain Requirement entries")
        if impl_ref in req.implementation_refs:
            self._update_status(req)
            return True
        req.implementation_refs.append(impl_ref)
        self._update_status(req)
        return True

    def link_verification(self, req_id: str, verif_ref: str) -> bool:
        """Link a verification artifact, returning false for an unknown ID."""
        if not isinstance(req_id, str) or not req_id.strip():
            raise ValueError("req_id must be a non-empty string")
        if not isinstance(verif_ref, str) or not verif_ref.strip():
            raise ValueError("verif_ref must be a non-empty string")
        req_id = req_id.strip()
        verif_ref = verif_ref.strip()
        req = self.requirements.get(req_id)
        if req is None:
            return False
        if not isinstance(req, Requirement):
            raise ValueError("requirements must contain Requirement entries")
        if verif_ref in req.verification_refs:
            self._update_status(req)
            return True
        req.verification_refs.append(verif_ref)
        self._update_status(req)
        return True

    def _update_status(self, req: Requirement) -> None:
        if not isinstance(req, Requirement):
            raise ValueError("req must be a Requirement")
        for impl_ref in req.implementation_refs:
            if not isinstance(impl_ref, str) or not impl_ref.strip():
                raise ValueError("implementation_refs must contain non-empty strings")
        for verif_ref in req.verification_refs:
            if not isinstance(verif_ref, str) or not verif_ref.strip():
                raise ValueError("verification_refs must contain non-empty strings")
        if req.implementation_refs and req.verification_refs:
            req.status = "verified"
        elif req.implementation_refs:
            req.status = "implemented"
        else:
            req.status = "open"

    @property
    def coverage(self) -> float:
        """Return the fraction of requirements with both link types."""
        if not self.requirements:
            return 0.0
        for req_key, req in self.requirements.items():
            if not isinstance(req, Requirement):
                raise ValueError("requirements must contain Requirement entries")
            if req.req_id != req_key:
                raise ValueError("requirement key mismatch with req_id")
            if not isinstance(req.status, str) or req.status not in {
                "open",
                "implemented",
                "verified",
            }:
                raise ValueError(
                    "requirements statuses must be one of: open, implemented, verified"
                )
            if not isinstance(req.standard, SafetyStandard):
                raise ValueError("requirements must use SafetyStandard")
            if not isinstance(req.sil_level, SILLevel):
                raise ValueError("requirements must use SILLevel")
        verified = sum(1 for r in self.requirements.values() if r.status == "verified")
        return verified / len(self.requirements)

    @property
    def open_count(self) -> int:
        """Return the number of requirements with no implementation link."""
        for req_key, req in self.requirements.items():
            if not isinstance(req, Requirement):
                raise ValueError("requirements must contain Requirement entries")
            if req.req_id != req_key:
                raise ValueError("requirement key mismatch with req_id")
            if not isinstance(req.status, str) or req.status not in {
                "open",
                "implemented",
                "verified",
            }:
                raise ValueError(
                    "requirements statuses must be one of: open, implemented, verified"
                )
        return sum(1 for r in self.requirements.values() if r.status == "open")

    @property
    def implemented_count(self) -> int:
        """Return the number with implementation but no verification link."""
        _ = self.coverage
        return sum(
            1 for requirement in self.requirements.values() if requirement.status == "implemented"
        )

    @property
    def verified_count(self) -> int:
        """Return the number with implementation and verification links."""
        _ = self.coverage
        return sum(
            1 for requirement in self.requirements.values() if requirement.status == "verified"
        )

    def generate_report(self, *, generated_at: str | None = None) -> str:
        """Generate a Markdown traceability report.

        The generated-at argument is an injection seam for reproducible
        packages. When omitted, an aware UTC timestamp is used.
        """
        for req_key, req in self.requirements.items():
            if not isinstance(req, Requirement):
                raise ValueError("requirements must contain Requirement entries")
            if req.req_id != req_key:
                raise ValueError("requirement key mismatch with req_id")
            if not isinstance(req.status, str) or req.status not in {
                "open",
                "implemented",
                "verified",
            }:
                raise ValueError(
                    "requirements statuses must be one of: open, implemented, verified"
                )
        timestamp = datetime.now(timezone.utc).isoformat() if generated_at is None else generated_at
        if not isinstance(timestamp, str) or not timestamp.strip():
            raise ValueError("generated_at must be a non-empty string when provided")
        lines = [
            "# Safety Traceability Matrix",
            f"Generated: {timestamp}",
            f"Coverage: {self.coverage:.1%} ({self.verified_count}/{len(self.requirements)})",
            "",
            "| Req ID | Standard | SIL | Status | Impl | Verif |",
            "|--------|----------|-----|--------|------|-------|",
        ]
        for req in sorted(self.requirements.values(), key=lambda r: r.req_id):
            lines.append(
                f"| {req.req_id} | {req.standard.value} | SIL {req.sil_level.value} "
                f"| {req.status} | {len(req.implementation_refs)} | {len(req.verification_refs)} |"
            )
        lines.extend(["", "## Artifact Links"])
        for req in sorted(self.requirements.values(), key=lambda r: r.req_id):
            implementation = ", ".join(
                reference.replace("\n", " ").replace("`", "\\`")
                for reference in req.implementation_refs
            )
            verification = ", ".join(
                reference.replace("\n", " ").replace("`", "\\`")
                for reference in req.verification_refs
            )
            lines.extend(
                [
                    f"### {req.req_id}",
                    f"- Implementation: {implementation or 'UNLINKED'}",
                    f"- Verification: {verification or 'UNLINKED'}",
                ]
            )
        return "\n".join(lines)


__all__ = [
    "Requirement",
    "TraceabilityMatrix",
]
