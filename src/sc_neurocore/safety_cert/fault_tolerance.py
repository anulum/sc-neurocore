# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Automated Safety & Regulatory Certification Generator

"""Common-cause and hardware-fault-tolerance assessments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List

from sc_neurocore.safety_cert.standards import SILLevel


@dataclass
class CCFDefence:
    """One defence against common cause failures."""

    defence_id: str
    description: str
    category: str  # "separation", "diversity", "complexity", "assessment", "competence"
    beta_reduction: float = 0.0  # Reduction in β-factor
    implemented: bool = False

    def __post_init__(self) -> None:
        """Validate one common-cause screening defence."""
        if not isinstance(self.defence_id, str) or not self.defence_id.strip():
            raise ValueError("defence_id must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.category, str) or self.category not in {
            "separation",
            "diversity",
            "complexity",
            "assessment",
            "competence",
        }:
            raise ValueError(
                "category must be one of: separation, diversity, complexity, assessment, competence"
            )
        if isinstance(self.beta_reduction, bool) or not isinstance(
            self.beta_reduction, int | float
        ):
            raise ValueError("beta_reduction must be a finite non-negative value")
        if not math.isfinite(float(self.beta_reduction)) or float(self.beta_reduction) < 0.0:
            raise ValueError("beta_reduction must be a finite non-negative value")
        if float(self.beta_reduction) > 1.0:
            raise ValueError("beta_reduction must not exceed 1.0")
        if not isinstance(self.implemented, bool):
            raise ValueError("implemented must be a boolean")


class CCFAnalysis:
    """Non-normative beta-factor screening helper.

    Default reductions are legacy modelling assumptions, not measured evidence
    or a conformity determination.
    """

    DEFAULT_DEFENCES = [
        CCFDefence("D1", "Physical separation of redundant channels", "separation", 0.025),
        CCFDefence("D2", "Diverse hardware implementations", "diversity", 0.02),
        CCFDefence("D3", "Diverse software / bitstream encodings", "diversity", 0.02),
        CCFDefence("D4", "Independent design teams", "competence", 0.01),
        CCFDefence("D5", "Environmental conditioning/shielding", "separation", 0.015),
        CCFDefence("D6", "Complexity analysis and minimisation", "complexity", 0.01),
    ]

    def __init__(self) -> None:
        for defence in self.DEFAULT_DEFENCES:
            if not isinstance(defence, CCFDefence):
                raise ValueError("DEFAULT_DEFENCES must contain CCFDefence entries")
        defence_ids = [defence.defence_id for defence in self.DEFAULT_DEFENCES]
        if len(defence_ids) != len(set(defence_ids)):
            raise ValueError("DEFAULT_DEFENCES must not contain duplicate defence_id values")
        self.defences: List[CCFDefence] = [
            CCFDefence(d.defence_id, d.description, d.category, d.beta_reduction, d.implemented)
            for d in self.DEFAULT_DEFENCES
        ]

    def mark_implemented(self, defence_id: str) -> bool:
        """Mark one known defence present, returning false for an unknown ID."""
        if not isinstance(defence_id, str) or not defence_id.strip():
            raise ValueError("defence_id must be a non-empty string")
        normalised_id = defence_id.strip()
        for d in self.defences:
            if not isinstance(d, CCFDefence):
                raise ValueError("defences must contain CCFDefence entries")
            if d.defence_id == normalised_id:
                d.implemented = True
                return True
        return False

    @property
    def beta_factor(self) -> float:
        """Resulting β-factor (start at 0.10, reduce by implemented defences)."""
        for defence in self.defences:
            if not isinstance(defence, CCFDefence):
                raise ValueError("defences must contain CCFDefence entries")
        base = 0.10
        reduction = 0.0
        for defence in self.defences:
            beta_reduction = float(defence.beta_reduction)
            if not math.isfinite(beta_reduction) or beta_reduction < 0.0 or beta_reduction > 1.0:
                raise ValueError("defence beta_reduction values must be finite values in [0, 1]")
            if defence.implemented:
                reduction += beta_reduction
        return max(0.005, base - reduction)

    @property
    def implemented_count(self) -> int:
        """Return the number of defences marked present."""
        count = 0
        for defence in self.defences:
            if not isinstance(defence, CCFDefence):
                raise ValueError("defences must contain CCFDefence entries")
            if defence.implemented:
                count += 1
        return count

    def sil_compatible(self, target_sil: SILLevel) -> bool:
        """Check if β is low enough for the target SIL."""
        if not isinstance(target_sil, SILLevel):
            raise ValueError("target_sil must be a SILLevel")
        thresholds = {
            SILLevel.SIL_1: 0.10,
            SILLevel.SIL_2: 0.05,
            SILLevel.SIL_3: 0.02,
            SILLevel.SIL_4: 0.01,
        }
        return self.beta_factor <= thresholds[target_sil]


class HFTLevel(Enum):
    """Hardware-fault-tolerance screening labels."""

    HFT_0 = 0
    HFT_1 = 1
    HFT_2 = 2


@dataclass
class HFTAssessment:
    """Hardware Fault Tolerance assessment per IEC 61508 Table 2."""

    sff: float
    target_sil: SILLevel

    def __post_init__(self) -> None:
        """Validate screening inputs."""
        if isinstance(self.sff, bool) or not isinstance(self.sff, int | float):
            raise ValueError("sff must be a finite value in [0, 1]")
        if not math.isfinite(float(self.sff)) or float(self.sff) < 0.0 or float(self.sff) > 1.0:
            raise ValueError("sff must be a finite value in [0, 1]")
        if not isinstance(self.target_sil, SILLevel):
            raise ValueError("target_sil must be a SILLevel")

    @property
    def required_hft(self) -> HFTLevel:
        """Determine required HFT from SFF and target SIL."""
        if not isinstance(self.target_sil, SILLevel):
            raise ValueError("target_sil must be a SILLevel")
        if not math.isfinite(float(self.sff)) or float(self.sff) < 0.0 or float(self.sff) > 1.0:
            raise ValueError("sff must be a finite value in [0, 1]")
        if self.sff >= 0.99:
            if self.target_sil.value <= 3:
                return HFTLevel.HFT_0
            return HFTLevel.HFT_1
        elif self.sff >= 0.90:
            if self.target_sil.value <= 2:
                return HFTLevel.HFT_0
            elif self.target_sil.value == 3:
                return HFTLevel.HFT_1
            return HFTLevel.HFT_2
        elif self.sff >= 0.60:
            if self.target_sil.value <= 1:
                return HFTLevel.HFT_0
            elif self.target_sil.value == 2:
                return HFTLevel.HFT_1
            return HFTLevel.HFT_2
        else:
            if self.target_sil.value <= 1:
                return HFTLevel.HFT_1
            return HFTLevel.HFT_2

    @property
    def is_simplex_ok(self) -> bool:
        """Return whether the legacy table screen yields HFT zero."""
        return self.required_hft == HFTLevel.HFT_0


__all__ = [
    "CCFDefence",
    "CCFAnalysis",
    "HFTLevel",
    "HFTAssessment",
]
