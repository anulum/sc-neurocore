# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Approximate computing modes

"""Precision-energy tradeoff configuration for approximate computing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ApproximationConfig:
    """Approximate computing configuration.

    Attributes
    ----------
    populations : dict[str, dict]
    total_energy_savings_pct : float
    max_output_error_pct : float
    """

    populations: dict[str, dict]
    total_energy_savings_pct: float
    max_output_error_pct: float


def configure_approximation(
    equations: dict[str, str],
    *,
    target_savings_pct: float = 30.0,
    max_error_pct: float = 5.0,
) -> ApproximationConfig:
    """Configure precision-energy tradeoff knobs per state variable."""
    pops = {}
    total_savings = 0.0
    for var in equations:
        bits_removable = min(4, int(target_savings_pct / 15))
        savings = bits_removable * 15.0 / len(equations)
        error = bits_removable * 1.5
        if error > max_error_pct:
            bits_removable = max(1, int(max_error_pct / 1.5))
            savings = bits_removable * 15.0 / len(equations)
            error = bits_removable * 1.5
        pops[var] = {
            "bits_reduced": bits_removable,
            "stochastic_rounding": bits_removable >= 2,
            "energy_savings_pct": round(savings, 1),
            "error_bound_pct": round(error, 2),
        }
        total_savings += savings

    return ApproximationConfig(
        populations=pops,
        total_energy_savings_pct=round(total_savings, 1),
        max_output_error_pct=max(p["error_bound_pct"] for p in pops.values()),
    )
