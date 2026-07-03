# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — User profile: chronotype and session preferences

"""User profile: chronotype and session preferences."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ── Chronotype ───────────────────────────────────────────────────────


class Chronotype(str, Enum):
    """Sleep chronotype model (after Dr. Michael Breus).

    Each chronotype has a preferred entrainment frequency range and
    optimal session timing.
    """

    LION = "lion"  # Early riser, alpha-dominant mornings
    BEAR = "bear"  # Solar schedule, balanced spectrum
    WOLF = "wolf"  # Night owl, theta-rich evenings
    DOLPHIN = "dolphin"  # Light sleeper, high beta baseline


# Default target Hz by chronotype
_CHRONOTYPE_TARGET_HZ: dict[Chronotype, float] = {
    Chronotype.LION: 10.0,  # Alpha (relaxed focus)
    Chronotype.BEAR: 10.0,  # Alpha (balanced)
    Chronotype.WOLF: 6.0,  # Theta (creative flow)
    Chronotype.DOLPHIN: 12.0,  # High alpha (calm alertness)
}

# Preferred cost weight profiles
_CHRONOTYPE_WEIGHTS: dict[Chronotype, dict[str, float]] = {
    Chronotype.LION: {
        "w_micro": 1.0,
        "w_reg": 0.01,
        "w_stability": 0.8,
    },
    Chronotype.BEAR: {
        "w_micro": 1.0,
        "w_reg": 0.01,
        "w_stability": 0.5,
    },
    Chronotype.WOLF: {
        "w_micro": 0.8,
        "w_reg": 0.02,
        "w_stability": 0.4,
    },
    Chronotype.DOLPHIN: {
        "w_micro": 1.2,
        "w_reg": 0.005,
        "w_stability": 1.0,
    },
}


# ── UserProfile ──────────────────────────────────────────────────────


@dataclass
class UserProfile:
    """Per-user preference and adaptation model.

    Parameters
    ----------
    user_id : str
        Unique user identifier.
    chronotype : Chronotype
        Sleep chronotype.
    baseline_band_powers : dict
        Resting-state EEG band powers (populated after first baseline).
    preferred_cost_weights : dict
        SSGF cost weights tuned to this user.
    sensitivity_map : dict
        Per-band sensitivity multipliers (e.g. {"alpha": 1.2}).
    session_count : int
        Total completed sessions.
    preferred_target_hz : float, optional
        Explicitly set target frequency (overrides chronotype default).
    """

    user_id: str = "anonymous"
    chronotype: Chronotype = Chronotype.BEAR
    baseline_band_powers: dict[str, float] = field(default_factory=dict)
    preferred_cost_weights: dict[str, float] = field(default_factory=dict)
    sensitivity_map: dict[str, float] = field(default_factory=dict)
    session_count: int = 0
    preferred_target_hz: float | None = None

    def __post_init__(self) -> None:
        """Populate chronotype-derived defaults for omitted profile maps."""
        # Populate defaults from chronotype if not provided
        if not self.preferred_cost_weights:
            self.preferred_cost_weights = dict(
                _CHRONOTYPE_WEIGHTS.get(self.chronotype, _CHRONOTYPE_WEIGHTS[Chronotype.BEAR]),
            )
        if not self.sensitivity_map:
            self.sensitivity_map = {
                "delta": 1.0,
                "theta": 1.0,
                "alpha": 1.0,
                "beta": 1.0,
                "gamma": 1.0,
            }

    # ── Target Hz ────────────────────────────────────────────────────

    def get_best_target_hz(self) -> float:
        """Return the best entrainment target for this user.

        Uses explicit preference if set, otherwise chronotype default.
        """
        if self.preferred_target_hz is not None:
            return self.preferred_target_hz
        return _CHRONOTYPE_TARGET_HZ.get(self.chronotype, 10.0)

    # ── Session Update ───────────────────────────────────────────────

    def update_from_session(
        self,
        avg_evs: float,
        peak_evs: float,
        best_target_hz: float | None = None,
        band_powers: dict[str, float] | None = None,
    ) -> None:
        """Update profile after a completed session.

        Parameters
        ----------
        avg_evs : float
            Average EVS score over the session.
        peak_evs : float
            Peak EVS score.
        best_target_hz : float, optional
            If the adaptive engine found a better target, adopt it.
        band_powers : dict, optional
            Updated baseline band powers from this session.
        """
        self.session_count += 1

        # Adopt best target if it outperformed
        if best_target_hz is not None and avg_evs > 50.0:
            if self.preferred_target_hz is None:
                self.preferred_target_hz = best_target_hz
            else:
                # Exponential moving average toward the new target
                alpha = 0.3
                self.preferred_target_hz = (
                    1 - alpha
                ) * self.preferred_target_hz + alpha * best_target_hz

        # Update baseline band powers (EMA blend)
        if band_powers:
            if not self.baseline_band_powers:
                self.baseline_band_powers = dict(band_powers)
            else:
                alpha = 0.2
                for band, power in band_powers.items():
                    old = self.baseline_band_powers.get(band, power)
                    self.baseline_band_powers[band] = (1 - alpha) * old + alpha * power

        logger.info(
            "Profile updated: session #%d, avg_evs=%.1f, target=%.2f Hz",
            self.session_count,
            avg_evs,
            self.preferred_target_hz or self.get_best_target_hz(),
        )

    # ── Serialisation ────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise the profile into JSON-compatible primitive values.

        Returns
        -------
        dict[str, Any]
            Snapshot containing the user identifier, chronotype value,
            per-band baselines, preferred SSGF cost weights, sensitivity map,
            completed session count, and optional preferred target frequency.
        """
        return {
            "user_id": self.user_id,
            "chronotype": self.chronotype.value,
            "baseline_band_powers": dict(self.baseline_band_powers),
            "preferred_cost_weights": dict(self.preferred_cost_weights),
            "sensitivity_map": dict(self.sensitivity_map),
            "session_count": self.session_count,
            "preferred_target_hz": self.preferred_target_hz,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserProfile:
        """Build a profile from a dictionary produced by :meth:`to_dict`.

        Parameters
        ----------
        data:
            JSON-compatible profile snapshot. Missing fields fall back to the
            same defaults used by the dataclass constructor.

        Returns
        -------
        UserProfile
            Reconstructed user profile with chronotype defaults populated in
            ``__post_init__`` when optional maps are absent.
        """
        chrono = data.get("chronotype", "bear")
        return cls(
            user_id=data.get("user_id", "anonymous"),
            chronotype=Chronotype(chrono),
            baseline_band_powers=data.get("baseline_band_powers", {}),
            preferred_cost_weights=data.get("preferred_cost_weights", {}),
            sensitivity_map=data.get("sensitivity_map", {}),
            session_count=data.get("session_count", 0),
            preferred_target_hz=data.get("preferred_target_hz"),
        )
