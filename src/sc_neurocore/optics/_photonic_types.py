# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Photonic value contracts

"""Typed value contracts shared by the photonic compiler responsibilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


def _require_finite(value: float, name: str) -> None:
    """Reject a non-finite numeric contract value."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _require_positive(value: float, name: str) -> None:
    """Reject a non-positive or non-finite numeric contract value."""
    _require_finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be strictly positive, got {value}")


def _require_non_negative(value: float, name: str) -> None:
    """Reject a negative or non-finite numeric contract value."""
    _require_finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")


class OpticalModulation(Enum):
    """Optical modulation scheme."""

    PHASE = "phase"
    AMPLITUDE = "amplitude"
    HYBRID = "hybrid"


@dataclass
class PhotonicTarget:
    """Hardware target specification for a photonic backend."""

    name: str
    wavelength_nm: float = 1550.0
    modulation: OpticalModulation = OpticalModulation.PHASE
    modulator_type: str = "MZI"
    q_factor: float = 15000.0
    insertion_loss_db: float = 0.5
    thermo_optic_coeff: float = 1.86e-4

    def __post_init__(self) -> None:
        """Validate target metadata before it reaches a compiler backend."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.modulator_type, str) or not self.modulator_type.strip():
            raise ValueError("modulator_type must be a non-empty string")
        if not isinstance(self.modulation, OpticalModulation):
            raise TypeError("modulation must be an OpticalModulation")
        _require_positive(self.wavelength_nm, "wavelength_nm")
        _require_positive(self.q_factor, "q_factor")
        _require_non_negative(self.insertion_loss_db, "insertion_loss_db")
        _require_finite(self.thermo_optic_coeff, "thermo_optic_coeff")

    @classmethod
    def lightmatter(cls) -> PhotonicTarget:
        """Return a Lightmatter-style photonic target profile."""
        return cls("Lightmatter", 1550.0, OpticalModulation.PHASE, "MZI", 20000.0, 0.3)

    @classmethod
    def silicon_photonics(cls) -> PhotonicTarget:
        """Return a generic silicon-photonics target profile."""
        return cls("SiPh-Generic", 1310.0, OpticalModulation.AMPLITUDE, "Microring", 12000.0, 0.8)

    @classmethod
    def two_d_waveguide(cls) -> PhotonicTarget:
        """Return a two-dimensional-material waveguide target profile."""
        return cls("2D-Material", 850.0, OpticalModulation.HYBRID, "MZI", 5000.0, 1.2)


@dataclass
class OpticalPulse:
    """Single optical pulse with phase and amplitude."""

    phase: float
    amplitude: float
    wavelength_nm: float
    duration_ps: float

    def __post_init__(self) -> None:
        """Validate the physical pulse boundary."""
        _require_finite(self.phase, "phase")
        _require_finite(self.amplitude, "amplitude")
        if not 0.0 <= self.amplitude <= 1.0:
            raise ValueError(f"amplitude must be in [0, 1], got {self.amplitude}")
        _require_positive(self.wavelength_nm, "wavelength_nm")
        _require_positive(self.duration_ps, "duration_ps")


__all__ = ["OpticalModulation", "OpticalPulse", "PhotonicTarget"]
