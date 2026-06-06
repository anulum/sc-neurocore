# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Model portability scorer

"""Score how portable a model is across all hardware profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PortabilityScore:
    """Cross-platform portability assessment.

    Attributes
    ----------
    score : float
        Portability score 0-100.
    compatible_profiles : int
        Number of compatible profiles.
    total_profiles : int
        Total profiles checked.
    blockers : list[str]
        Portability blockers.
    """

    score: float
    compatible_profiles: int
    total_profiles: int
    blockers: list[str]


def score_portability(
    equations: dict[str, str],
    *,
    min_data_width: int = 8,
) -> PortabilityScore:
    """Score how portable a model is across all profiles."""
    from ...platforms import (
        list_profile_names,
        get_profile,
    )

    total_ops = sum(e.count("*") + e.count("/") for e in equations.values())
    names = list_profile_names()
    compatible = 0
    blockers = []

    for n in names:
        p = get_profile(n)
        if p.data_width < min_data_width:
            continue
        if (
            total_ops > 3
            and not p.dsp_block
            and p.platform_class
            not in (
                "simulation",
                "biological",
                "dna_molecular",
            )
        ):
            continue
        compatible += 1

    if total_ops > 5:
        blockers.append("High arithmetic complexity limits low-width targets")
    if len(equations) > 4:
        blockers.append("Many state variables require large register files")

    pct = (compatible / len(names)) * 100 if names else 0
    return PortabilityScore(
        score=round(pct, 1),
        compatible_profiles=compatible,
        total_profiles=len(names),
        blockers=blockers,
    )
