# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SEU scrub scheduler

"""Optimal scrubbing interval calculation for space-grade FPGA configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScrubSchedule:
    """Configuration memory scrubbing schedule.

    Attributes
    ----------
    interval_ms : float
    strategy : str
    frames_per_cycle : int
    expected_seu_rate : float
    """

    interval_ms: float
    strategy: str
    frames_per_cycle: int
    expected_seu_rate: float


def schedule_seu_scrubbing(
    config_bits: int,
    *,
    orbit_altitude_km: float = 400.0,
    shielding_mm_al: float = 3.0,
    strategy: str = "hybrid",
) -> ScrubSchedule:
    """Generate scrubbing schedule for space-grade configuration memory."""
    base_rate = 1e-7
    altitude_factor = orbit_altitude_km / 400.0
    shielding_factor = max(0.1, 1.0 - shielding_mm_al * 0.15)
    seu_rate = base_rate * altitude_factor * shielding_factor

    expected_upsets_per_day = seu_rate * config_bits
    if expected_upsets_per_day > 0:
        interval_hours = 1.0 / expected_upsets_per_day
    else:
        interval_hours = 24.0
    interval_ms = interval_hours * 3_600_000

    frame_size = 1024
    frames = max(1, config_bits // frame_size)

    return ScrubSchedule(
        interval_ms=round(interval_ms, 2),
        strategy=strategy,
        frames_per_cycle=frames,
        expected_seu_rate=round(expected_upsets_per_day, 6),
    )
