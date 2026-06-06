# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Target recommender

"""Hardware target recommendation utilities.

Ranks all registered hardware profiles based on model complexity and
user constraints (power, frequency, width).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetRecommendation:
    """Ranked hardware target recommendation.

    Attributes
    ----------
    profile_name : str
        Recommended profile.
    score : float
        Fitness score (0-100).
    rationale : str
        Why this target is recommended.
    """

    profile_name: str
    score: float
    rationale: str


def recommend_target(
    equations: dict[str, str],
    *,
    max_power_mw: float | None = None,
    min_freq_mhz: float | None = None,
    max_data_width: int | None = None,
    require_class: str | None = None,
    top_n: int = 5,
) -> list[TargetRecommendation]:
    """Recommend optimal hardware targets for a neuron model.

    Given ODE equations and constraints, ranks all registered
    profiles and returns the top N recommendations.

    Parameters
    ----------
    equations : dict[str, str]
        ODE equations.
    max_power_mw : float, optional
        Maximum power budget.
    min_freq_mhz : float, optional
        Minimum clock frequency.
    max_data_width : int, optional
        Maximum data width.
    require_class : str, optional
        Required platform class.
    top_n : int
        Number of recommendations.

    Returns
    -------
    list[TargetRecommendation]
        Ranked recommendations.
    """
    from ..platforms import (
        list_profile_names,
        get_profile,
    )

    # Count operations for complexity
    total_ops = sum(
        e.count("+") + e.count("-") + e.count("*") + e.count("/") for e in equations.values()
    )
    num_vars = len(equations)

    scored = []
    for name in list_profile_names():
        p = get_profile(name)
        score = 50.0  # baseline

        # Class filter
        if require_class and p.platform_class != require_class:
            continue

        # Width filter
        if max_data_width and p.data_width > max_data_width:
            continue

        # Frequency constraint
        if min_freq_mhz and p.max_freq_mhz and p.max_freq_mhz < min_freq_mhz:
            continue

        # Scoring: prefer wider data for complex models
        if total_ops > 5 and p.data_width >= 16:
            score += 15
        elif total_ops <= 5 and p.data_width <= 16:
            score += 10

        # DSP availability bonus
        if p.dsp_block:
            score += 10

        # Frequency bonus
        if p.max_freq_mhz and p.max_freq_mhz > 500:
            score += 5

        # Neuromorphic bonus for SNN
        if p.platform_class in ("neuromorphic", "biological"):
            score += 10

        # Edge bonus for simple models
        if num_vars <= 2 and p.platform_class in ("edge_mcu", "analog_mixed"):
            score += 10

        rationale = (
            f"{p.vendor} {p.family}: Q{p.data_width - p.fraction}.{p.fraction}, {p.platform_class}"
        )
        if p.max_freq_mhz:
            rationale += f", {p.max_freq_mhz} MHz"

        scored.append(
            TargetRecommendation(
                profile_name=name,
                score=round(score, 1),
                rationale=rationale,
            )
        )

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_n]
