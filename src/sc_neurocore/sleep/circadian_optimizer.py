"""
Circadian Optimizer — Chronotype-based circadian profiling
============================================================

Supports 4 chronotypes with optimal sleep/wake timing and
protocol recommendations.

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np


class Chronotype(str, Enum):
    """Circadian chronotype."""
    LION = "lion"       # Early bird
    BEAR = "bear"       # Average
    WOLF = "wolf"       # Night owl
    DOLPHIN = "dolphin" # Light sleeper


@dataclass
class CircadianProfile:
    """Circadian timing profile for a chronotype."""
    chronotype: Chronotype
    optimal_bedtime_h: float      # 24h format
    optimal_wake_h: float
    melatonin_onset_h: float      # DLMO (dim light melatonin onset)
    core_body_temp_nadir_h: float  # Lowest core temperature
    optimal_nap_h: float
    sleep_pressure_peak_h: float
    recommended_protocol: str


CIRCADIAN_PROFILES = {
    Chronotype.LION: CircadianProfile(
        chronotype=Chronotype.LION,
        optimal_bedtime_h=21.5,
        optimal_wake_h=5.5,
        melatonin_onset_h=20.0,
        core_body_temp_nadir_h=3.5,
        optimal_nap_h=13.0,
        sleep_pressure_peak_h=21.0,
        recommended_protocol="deep_sleep_boost",
    ),
    Chronotype.BEAR: CircadianProfile(
        chronotype=Chronotype.BEAR,
        optimal_bedtime_h=23.0,
        optimal_wake_h=7.0,
        melatonin_onset_h=21.5,
        core_body_temp_nadir_h=4.5,
        optimal_nap_h=14.0,
        sleep_pressure_peak_h=22.5,
        recommended_protocol="insomnia_relief",
    ),
    Chronotype.WOLF: CircadianProfile(
        chronotype=Chronotype.WOLF,
        optimal_bedtime_h=0.5,
        optimal_wake_h=8.5,
        melatonin_onset_h=23.0,
        core_body_temp_nadir_h=6.0,
        optimal_nap_h=15.5,
        sleep_pressure_peak_h=0.0,
        recommended_protocol="rem_enhancement",
    ),
    Chronotype.DOLPHIN: CircadianProfile(
        chronotype=Chronotype.DOLPHIN,
        optimal_bedtime_h=23.5,
        optimal_wake_h=6.5,
        melatonin_onset_h=22.0,
        core_body_temp_nadir_h=4.0,
        optimal_nap_h=13.5,
        sleep_pressure_peak_h=23.0,
        recommended_protocol="insomnia_relief",
    ),
}


class CircadianOptimizer:
    """
    Provides circadian-optimized timing and protocol recommendations.
    """

    def __init__(self, chronotype: Chronotype = Chronotype.BEAR):
        self.chronotype = chronotype
        self.profile = CIRCADIAN_PROFILES[chronotype]

    def get_profile(self) -> CircadianProfile:
        """Return the circadian profile."""
        return self.profile

    def get_sleep_window(self) -> Dict[str, float]:
        """Return optimal sleep/wake window."""
        return {
            "bedtime_h": self.profile.optimal_bedtime_h,
            "wake_h": self.profile.optimal_wake_h,
            "duration_h": self._sleep_duration(),
        }

    def _sleep_duration(self) -> float:
        """Compute sleep duration accounting for midnight crossover."""
        wake = self.profile.optimal_wake_h
        bed = self.profile.optimal_bedtime_h
        if wake < bed:
            return 24.0 - bed + wake
        return wake - bed

    def get_recommended_protocol(self) -> str:
        """Return recommended sleep protocol name."""
        return self.profile.recommended_protocol

    def is_in_sleep_window(self, hour: float) -> bool:
        """Check if given hour (24h format) is within sleep window."""
        bed = self.profile.optimal_bedtime_h
        wake = self.profile.optimal_wake_h
        if bed > wake:
            # Crosses midnight
            return hour >= bed or hour <= wake
        return bed <= hour <= wake

    def melatonin_level(self, hour: float) -> float:
        """Estimate relative melatonin level (0-1) at given hour."""
        onset = self.profile.melatonin_onset_h
        # Peak ~3h after onset, decline ~2h before wake
        peak_h = (onset + 3) % 24
        # Simple sinusoidal model
        phase = (hour - onset) % 24
        if phase < 0:
            phase += 24
        if phase <= 8:  # Rising and peak period
            return float(np.clip(np.sin(np.pi * phase / 8.0), 0, 1))
        elif phase <= 12:
            return float(np.clip(np.cos(np.pi * (phase - 8) / 8.0), 0, 1))
        return 0.0

    def to_dict(self) -> Dict:
        """Return profile as dict."""
        p = self.profile
        return {
            "chronotype": p.chronotype.value,
            "optimal_bedtime_h": p.optimal_bedtime_h,
            "optimal_wake_h": p.optimal_wake_h,
            "melatonin_onset_h": p.melatonin_onset_h,
            "sleep_duration_h": self._sleep_duration(),
            "recommended_protocol": p.recommended_protocol,
        }
