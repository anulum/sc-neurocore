"""
User Profile — Personalization priors for adaptive audio
=========================================================

Stores:
- Chronotype (circadian preference)
- Baseline EEG band powers
- Per-band sensitivity (responsiveness from prior sessions)
- Preferred SSGF cost weights (learned over time)

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import numpy as np


class Chronotype(str, Enum):
    """Circadian chronotype (from CircadianOptimizer)."""
    LION = "lion"       # Early bird (4-6am wake)
    BEAR = "bear"       # Average (7am wake)
    WOLF = "wolf"       # Night owl (9-10am wake)
    DOLPHIN = "dolphin" # Light sleeper (variable)


@dataclass
class UserProfile:
    """User priors for adaptive audio personalization."""

    user_id: str = "default"
    chronotype: Chronotype = Chronotype.BEAR

    # Baseline EEG from EVS
    baseline_band_powers: Dict[str, float] = field(default_factory=lambda: {
        "delta": 0.0, "theta": 0.0, "alpha": 0.0, "beta": 0.0, "gamma": 0.0,
    })

    # Per-band sensitivity (how responsive this user is to each band)
    # Higher = more responsive, learned from session history
    sensitivity_map: Dict[str, float] = field(default_factory=lambda: {
        "delta": 0.5, "theta": 0.5, "alpha": 0.5, "beta": 0.5, "gamma": 0.5,
    })

    # Preferred SSGF cost weights (from best sessions)
    preferred_cost_weights: Dict[str, float] = field(default_factory=lambda: {
        "w_micro": 1.0, "w_spectral": 0.5, "w_reg": 0.1,
    })

    # Session history: list of (target_hz, evs_peak, evs_avg)
    session_count: int = 0
    best_evs_score: float = 0.0

    def get_optimal_target_hz(self) -> float:
        """Suggest optimal entrainment frequency based on chronotype."""
        # Chronotype-based defaults
        defaults = {
            Chronotype.LION: 10.0,     # Alpha for morning alertness
            Chronotype.BEAR: 10.0,     # Standard alpha
            Chronotype.WOLF: 8.0,      # Lower alpha/high theta for evening
            Chronotype.DOLPHIN: 12.0,  # Higher alpha for focus
        }
        base = defaults.get(self.chronotype, 10.0)

        # Adjust based on sensitivity (prefer bands user responds to)
        max_sensitivity_band = max(self.sensitivity_map, key=self.sensitivity_map.get)
        band_centers = {"delta": 2.0, "theta": 6.0, "alpha": 10.5, "beta": 20.0, "gamma": 40.0}
        if self.sensitivity_map[max_sensitivity_band] > 0.7:
            # Blend toward most responsive band
            responsive_hz = band_centers.get(max_sensitivity_band, 10.0)
            base = 0.7 * base + 0.3 * responsive_hz

        return round(base, 1)

    def get_ssgf_config_overrides(self) -> Dict[str, float]:
        """Get SSGF config overrides based on profile."""
        overrides = dict(self.preferred_cost_weights)

        # Chronotype adjustments
        if self.chronotype == Chronotype.WOLF:
            overrides["sigma_g"] = 0.4  # Stronger geometry for night owls
            overrides["lr_z"] = 0.015
        elif self.chronotype == Chronotype.DOLPHIN:
            overrides["sigma_g"] = 0.2  # Gentler for light sleepers
            overrides["noise_std"] = 0.03

        return overrides

    def update_from_session(self, target_hz: float, evs_avg: float, evs_peak: float):
        """Update profile from completed session."""
        self.session_count += 1
        if evs_peak > self.best_evs_score:
            self.best_evs_score = evs_peak

        # Update sensitivity for the target band
        for band, (lo, hi) in [
            ("delta", (0.5, 4)), ("theta", (4, 8)),
            ("alpha", (8, 13)), ("beta", (13, 30)), ("gamma", (30, 100)),
        ]:
            if lo <= target_hz <= hi:
                # Exponential moving average of responsiveness
                old = self.sensitivity_map.get(band, 0.5)
                responsiveness = evs_avg / 100.0  # Normalize EVS to [0, 1]
                self.sensitivity_map[band] = 0.8 * old + 0.2 * responsiveness
                break

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "chronotype": self.chronotype.value,
            "baseline_band_powers": self.baseline_band_powers,
            "sensitivity_map": self.sensitivity_map,
            "preferred_cost_weights": self.preferred_cost_weights,
            "session_count": self.session_count,
            "best_evs_score": self.best_evs_score,
            "optimal_target_hz": self.get_optimal_target_hz(),
        }
