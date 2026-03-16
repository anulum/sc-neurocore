# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations
from typing import Any

"""Circadian rhythm modelling and chronotype-aware sleep optimisation.

Four chronotypes (Lion, Bear, Wolf, Dolphin) each carry a preferred
bed-/wake-time and a default audio protocol.  The optimizer provides
melatonin-level estimation (sinusoidal model) and protocol recommendations.
"""


import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Chronotypes
# ---------------------------------------------------------------------------


class Chronotype(Enum):
    """Sleep chronotype labels (after Michael Breus' model)."""

    LION = "lion"
    BEAR = "bear"
    WOLF = "wolf"
    DOLPHIN = "dolphin"


# ---------------------------------------------------------------------------
# Circadian profile
# ---------------------------------------------------------------------------


@dataclass
class CircadianProfile:
    """A chronotype's circadian parameters.

    Attributes
    ----------
    chronotype : Chronotype
        The chronotype label.
    bedtime_hour : float
        Ideal bedtime in decimal hours (0-24, can exceed 24 for next-day).
    wake_hour : float
        Ideal wake time in decimal hours.
    default_protocol : str
        Name of the recommended sleep audio protocol.
    melatonin_peak_hour : float
        Hour at which endogenous melatonin peaks (typically ~2 h after
        bedtime onset for most chronotypes).
    core_body_temp_nadir_hour : float
        Hour of the core body temperature nadir (~2 h after melatonin peak).
    """

    chronotype: Chronotype
    bedtime_hour: float
    wake_hour: float
    default_protocol: str
    melatonin_peak_hour: float
    core_body_temp_nadir_hour: float


# ---------------------------------------------------------------------------
# Pre-built profiles
# ---------------------------------------------------------------------------

_PROFILES: Dict[Chronotype, CircadianProfile] = {
    Chronotype.LION: CircadianProfile(
        chronotype=Chronotype.LION,
        bedtime_hour=21.5,
        wake_hour=5.5,
        default_protocol="deep_sleep_boost",
        melatonin_peak_hour=23.5,
        core_body_temp_nadir_hour=1.5,
    ),
    Chronotype.BEAR: CircadianProfile(
        chronotype=Chronotype.BEAR,
        bedtime_hour=23.0,
        wake_hour=7.0,
        default_protocol="insomnia_relief",
        melatonin_peak_hour=1.0,
        core_body_temp_nadir_hour=3.0,
    ),
    Chronotype.WOLF: CircadianProfile(
        chronotype=Chronotype.WOLF,
        bedtime_hour=0.5,
        wake_hour=8.5,
        default_protocol="rem_enhancement",
        melatonin_peak_hour=2.5,
        core_body_temp_nadir_hour=4.5,
    ),
    Chronotype.DOLPHIN: CircadianProfile(
        chronotype=Chronotype.DOLPHIN,
        bedtime_hour=23.5,
        wake_hour=6.5,
        default_protocol="insomnia_relief",
        melatonin_peak_hour=1.5,
        core_body_temp_nadir_hour=3.5,
    ),
}


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class CircadianOptimizer:
    """Chronotype-aware circadian rhythm optimizer.

    Parameters
    ----------
    chronotype : Chronotype
        The user's chronotype.

    Example::

        opt = CircadianOptimizer(Chronotype.BEAR)
        profile = opt.get_profile()
        print(opt.melatonin_level(23.0))  # near-peak for a Bear
    """

    def __init__(self, chronotype: Chronotype) -> None:
        self.chronotype = chronotype
        self._profile = _PROFILES[chronotype]

    # -- public API ---------------------------------------------------------

    def get_profile(self) -> CircadianProfile:
        """Return the full circadian profile for the configured chronotype."""
        return self._profile

    def get_sleep_window(self) -> Tuple[float, float]:
        """Return ``(bedtime_hour, wake_hour)``."""
        return (self._profile.bedtime_hour, self._profile.wake_hour)

    def get_recommended_protocol(self) -> str:
        """Return the default protocol name for this chronotype."""
        return self._profile.default_protocol

    def is_in_sleep_window(self, hour: float) -> bool:
        """Check whether *hour* (0-24) falls inside the sleep window.

        Handles windows that wrap past midnight (e.g. 23.5 -> 6.5).
        """
        bed = self._profile.bedtime_hour
        wake = self._profile.wake_hour

        if bed <= wake:
            return bed <= hour < wake
        else:
            # wraps past midnight
            return hour >= bed or hour < wake

    def melatonin_level(self, hour: float) -> float:
        """Estimate melatonin level at *hour* using a sinusoidal model.

        Returns a value in ``[0, 1]`` where 1 is the peak (at
        ``melatonin_peak_hour``) and 0 is the trough (12 h offset).
        """
        peak = self._profile.melatonin_peak_hour
        # phase so that cos(0) = 1 at the peak hour
        phase = 2.0 * math.pi * (hour - peak) / 24.0
        level = 0.5 * (1.0 + math.cos(phase))
        return float(np.clip(level, 0.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the optimizer state to a plain dict."""
        p = self._profile
        return {
            "chronotype": self.chronotype.value,
            "bedtime_hour": p.bedtime_hour,
            "wake_hour": p.wake_hour,
            "default_protocol": p.default_protocol,
            "melatonin_peak_hour": p.melatonin_peak_hour,
            "core_body_temp_nadir_hour": p.core_body_temp_nadir_hour,
        }
