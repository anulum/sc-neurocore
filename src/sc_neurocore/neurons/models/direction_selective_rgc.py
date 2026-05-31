# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Direction-Selective Retinal Ganglion Cell

"""Direction-selective retinal ganglion cell with exact membrane relaxation.

The model implements On/Off centre response, low-pass surround inhibition, and
an exact first-order membrane update for the drive held constant over one step.
Invalid optical drive or corrupted runtime state fails before mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class DirectionSelectiveRGC:
    """Direction-selective retinal ganglion cell.

    Parameters are finite and physical: positive ``tau``, ``theta`` and ``dt``;
    non-negative centre/surround weights; finite preferred direction and state.
    """

    tau: float = 10.0
    theta: float = 0.5
    is_on_centre: bool = True
    w_centre: float = 1.0
    w_surround: float = 0.3
    direction_pref: float = 0.0
    dt: float = 1.0

    v: float = 0.0
    _prev_intensity: float = 0.0
    _surround: float = 0.0

    def __post_init__(self) -> None:
        for name in ("tau", "theta", "dt"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, float(value))

        if type(self.is_on_centre) is not bool:
            raise ValueError("is_on_centre must be bool")

        for name in ("w_centre", "w_surround", "_prev_intensity", "_surround"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            setattr(self, name, float(value))

        for name in ("direction_pref", "v"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, float(value))

    @classmethod
    def new_on(cls) -> DirectionSelectiveRGC:
        """Create an On-centre cell."""
        return cls(is_on_centre=True)

    @classmethod
    def new_off(cls) -> DirectionSelectiveRGC:
        """Create an Off-centre cell."""
        return cls(is_on_centre=False)

    @staticmethod
    def _finite_non_negative(name: str, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite and non-negative")
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return value

    def _validate_runtime(self) -> None:
        if not math.isfinite(self.v):
            raise ValueError("v must be finite")
        for name in ("tau", "theta", "dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("w_centre", "w_surround", "_prev_intensity", "_surround"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if type(self.is_on_centre) is not bool:
            raise ValueError("is_on_centre must be bool")

    def step_rf(self, intensity: float, surround_mean: float) -> int:
        """Step with local intensity and surround mean intensity.

        Returns 1 if the exact membrane candidate crosses threshold, otherwise
        0. The temporal buffers are committed only after all candidate state is
        finite and physically valid.
        """
        intensity = self._finite_non_negative("intensity", intensity)
        surround_mean = self._finite_non_negative("surround_mean", surround_mean)
        self._validate_runtime()

        temporal_diff = intensity - self._prev_intensity
        centre_response = self.w_centre * temporal_diff if self.is_on_centre else -self.w_centre * temporal_diff
        next_surround = 0.9 * self._surround + 0.1 * surround_mean
        surround_inhib = self.w_surround * next_surround
        drive = centre_response - surround_inhib
        decay = math.exp(-self.dt / self.tau)
        next_v = drive + (self.v - drive) * decay

        if not all(math.isfinite(value) for value in (next_surround, drive, decay, next_v)) or next_surround < 0.0:
            raise ValueError("DirectionSelectiveRGC candidate state must be finite and physical")

        self._prev_intensity = intensity
        self._surround = next_surround
        if next_v >= self.theta:
            self.v = 0.0
            return 1
        self.v = next_v
        return 0

    def step(self, current: float) -> int:
        """Simple step with no surround input."""
        return self.step_rf(current, 0.0)

    def reset(self) -> None:
        """Reset state to initial conditions."""
        self.v = 0.0
        self._prev_intensity = 0.0
        self._surround = 0.0
