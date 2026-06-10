# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adiabatic clock generator

"""Generate multi-phase clocking for adiabatic (reversible) computing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdiabaticPhase:
    """Adiabatic clock phase timing (ps).

    Attributes
    ----------
    name : str
    rise_ps : float
    hold_ps : float
    fall_ps : float
    sleep_ps : float
    """

    name: str
    rise_ps: float
    hold_ps: float
    fall_ps: float
    sleep_ps: float


def generate_adiabatic_clocks(phases: int, freq_mhz: float) -> list[AdiabaticPhase]:
    """Generate multi-phase clocking for adiabatic computing."""
    period_ps = 1_000_000.0 / freq_mhz

    segment_ps = period_ps / 4.0

    clock_schedule = []
    for i in range(phases):
        clock_schedule.append(
            AdiabaticPhase(
                name=f"PHI_{i}",
                rise_ps=round(segment_ps, 1),
                hold_ps=round(segment_ps, 1),
                fall_ps=round(segment_ps, 1),
                sleep_ps=round(segment_ps, 1),
            )
        )
    return clock_schedule
