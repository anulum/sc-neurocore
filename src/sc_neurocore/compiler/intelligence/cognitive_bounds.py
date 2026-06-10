# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cognitive bound enforcer

"""Enforce safe operating ranges on cognitive models via RTL clamping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CognitiveBounds:
    """Clamping results for cognitive stability.

    Attributes
    ----------
    safe_equations : dict[str, str]
    lyapunov_divergence_proxy : float
    switches_inserted : int
    """

    safe_equations: dict[str, str]
    lyapunov_divergence_proxy: float
    switches_inserted: int


def enforce_cognitive_bounds(
    equations: dict[str, str],
    state_bounds: dict[str, tuple[float, float]],
) -> CognitiveBounds:
    """Enforce safe operating ranges on cognitive models via RTL clamping."""
    safe_eqs = {}
    switches = 0
    lyapunov = 0.0

    for var, expr in equations.items():
        if var in state_bounds:
            min_v, max_v = state_bounds[var]
            safe_eqs[var] = (
                f"({expr}) > {max_v} ? {max_v} : (({expr}) < {min_v} ? {min_v} : ({expr}))"
            )
            switches += 2
            lyapunov += abs(max_v - min_v) / 100.0
        else:
            safe_eqs[var] = expr

    return CognitiveBounds(
        safe_equations=safe_eqs,
        lyapunov_divergence_proxy=round(lyapunov, 4),
        switches_inserted=switches,
    )
