# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Automated Safety & Regulatory Certification Generator

"""Safety change-impact records and re-verification tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ChangeRecord:
    """One tracked change affecting safety."""

    change_id: str
    description: str
    affected_modules: List[str]
    affected_reqs: List[str]
    risk_level: str = "low"  # "low", "medium", "high"
    re_verification_needed: bool = False

    def __post_init__(self) -> None:
        """Validate one change record and its affected identifiers."""
        if not isinstance(self.change_id, str) or not self.change_id.strip():
            raise ValueError("change_id must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.risk_level, str) or self.risk_level not in {"low", "medium", "high"}:
            raise ValueError("risk_level must be one of: low, medium, high")
        if not isinstance(self.re_verification_needed, bool):
            raise ValueError("re_verification_needed must be a boolean")
        if not isinstance(self.affected_modules, list):
            raise ValueError("affected_modules must be a list")
        if not isinstance(self.affected_reqs, list):
            raise ValueError("affected_reqs must be a list")

        for module in self.affected_modules:
            if not isinstance(module, str) or not module.strip():
                raise ValueError("affected_modules must contain non-empty strings")
            if module != module.strip():
                raise ValueError("affected_modules must not contain surrounding whitespace")
        for req in self.affected_reqs:
            if not isinstance(req, str) or not req.strip():
                raise ValueError("affected_reqs must contain non-empty strings")
            if req != req.strip():
                raise ValueError("affected_reqs must not contain surrounding whitespace")
        if len(self.affected_modules) != len(set(self.affected_modules)):
            raise ValueError("affected_modules must not contain duplicates")
        if len(self.affected_reqs) != len(set(self.affected_reqs)):
            raise ValueError("affected_reqs must not contain duplicates")


class ChangeImpactTracker:
    """Tracks changes and their impact on certification artifacts."""

    def __init__(self) -> None:
        self.changes: List[ChangeRecord] = []

    def add_change(self, change: ChangeRecord) -> None:
        """Add one unique change and derive its re-verification flag."""
        if not isinstance(change, ChangeRecord):
            raise ValueError("change must be a ChangeRecord")
        if any(existing.change_id == change.change_id for existing in self.changes):
            raise ValueError("change_id values must be unique")
        if change.risk_level in ("medium", "high"):
            change.re_verification_needed = True
        self.changes.append(change)

    def affected_requirements(self) -> List[str]:
        """Return sorted unique requirement IDs affected by recorded changes."""
        reqs: set[str] = set()
        for c in self.changes:
            if not isinstance(c, ChangeRecord):
                raise ValueError("changes must contain ChangeRecord entries")
            if not isinstance(c.change_id, str) or not c.change_id.strip():
                raise ValueError("change_id values must be non-empty strings")
            if not isinstance(c.affected_reqs, list):
                raise ValueError("affected_reqs must be a list")
            for req in c.affected_reqs:
                if not isinstance(req, str) or not req.strip():
                    raise ValueError("affected_reqs must contain non-empty strings")
            reqs.update(c.affected_reqs)
        return sorted(reqs)

    @property
    def high_risk_count(self) -> int:
        """Return the number of changes labelled high risk."""
        count = 0
        for change in self.changes:
            if not isinstance(change, ChangeRecord):
                raise ValueError("changes must contain ChangeRecord entries")
            if not isinstance(change.risk_level, str) or change.risk_level not in {
                "low",
                "medium",
                "high",
            }:
                raise ValueError("changes risk_level values must be one of: low, medium, high")
            if change.risk_level == "high":
                count += 1
        return count

    @property
    def needs_re_certification(self) -> bool:
        """Return the legacy screen for at least one high-risk change."""
        return self.high_risk_count > 0


__all__ = [
    "ChangeRecord",
    "ChangeImpactTracker",
]
