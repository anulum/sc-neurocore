# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cochlear Inner Hair Cell (Zilany et al. 2009)

"""Cochlear inner hair cell: mechano-electrical transduction.

Converts basilar membrane displacement (mechanical) into receptor potential
via stereocilia tip-link channels with Boltzmann activation:

    P_open(x) = 1 / (1 + exp(-(x - x_0) / delta))
    I_MET = g_max * P_open * (V - E_MET)
    C * dV/dt = -g_L * (V - E_L) - I_MET + I_ext

Glutamate release is graded (not spiked): proportional to depolarisation
above resting potential. For compatibility with the spike-based framework,
``step()`` returns 1 if glutamate_release > 0.5, else 0.

Reference: Meddis (2006), Zilany et al. (2009, 2014).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CochlearHairCell:
    """Cochlear inner hair cell with Boltzmann MET channel activation.

    Parameters
    ----------
    g_max : float
        Maximum MET channel conductance. Default: 10.0.
    e_met : float
        MET channel reversal potential (mV). Default: 0.0.
    g_l : float
        Leak conductance. Default: 1.0.
    e_l : float
        Leak reversal / resting potential (mV). Default: -60.0.
    cap : float
        Membrane capacitance (pF). Default: 10.0.
    x0 : float
        Boltzmann half-activation displacement (nm). Default: 0.0.
    delta : float
        Boltzmann slope factor (nm). Default: 0.1.
    dt : float
        Integration timestep (ms). Default: 0.01.
    """

    g_max: float = 10.0
    e_met: float = 0.0
    g_l: float = 1.0
    e_l: float = -60.0
    cap: float = 10.0
    x0: float = 0.0
    delta: float = 0.1
    dt: float = 0.01

    v: float = -60.0
    glutamate_release: float = 0.0

    def __post_init__(self) -> None:
        self._validate_runtime_state()

    def _validate_runtime_state(self) -> None:
        for name in ("g_l", "cap", "delta", "dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        if not math.isfinite(self.g_max) or self.g_max < 0.0:
            raise ValueError("g_max must be finite and non-negative")

        for name in ("e_met", "e_l", "x0", "v"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if not math.isfinite(self.glutamate_release) or self.glutamate_release < 0.0:
            raise ValueError("glutamate_release must be finite and non-negative")

    def p_open(self, displacement: float) -> float:
        """Boltzmann activation of MET channels."""
        if not math.isfinite(displacement):
            raise ValueError("displacement must be finite")
        if not math.isfinite(self.x0):
            raise ValueError("x0 must be finite")
        if not math.isfinite(self.delta) or self.delta <= 0.0:
            raise ValueError("delta must be finite and positive")

        z = (displacement - self.x0) / self.delta
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def step(self, displacement: float) -> int:
        """Step with basilar membrane displacement.

        Returns 1 if glutamate_release > 0.5, else 0.
        """
        self._validate_runtime_state()
        po = self.p_open(displacement)
        g_met = self.g_max * po
        g_total = self.g_l + g_met
        if not math.isfinite(g_total) or g_total <= 0.0:
            raise FloatingPointError("total cochlear conductance must be finite and positive")
        v_inf = (self.g_l * self.e_l + g_met * self.e_met) / g_total
        candidate_v = v_inf + (self.v - v_inf) * math.exp(-(g_total / self.cap) * self.dt)
        if not math.isfinite(candidate_v):
            raise FloatingPointError("cochlear voltage candidate must be finite")
        candidate_release = max(candidate_v + 60.0, 0.0) / 40.0
        if not math.isfinite(candidate_release):
            raise FloatingPointError("cochlear glutamate release candidate must be finite")

        self.v = candidate_v

        # Graded glutamate release proportional to depolarisation.
        self.glutamate_release = candidate_release
        return 1 if self.glutamate_release > 0.5 else 0

    def reset(self) -> None:
        """Reset state to resting potential."""
        self.v = self.e_l
        self.glutamate_release = 0.0

    def state(self) -> dict[str, float]:
        """Return a compact state and parameter snapshot for reproducibility."""
        return {
            "g_max": self.g_max,
            "e_met": self.e_met,
            "g_l": self.g_l,
            "e_l": self.e_l,
            "cap": self.cap,
            "x0": self.x0,
            "delta": self.delta,
            "dt": self.dt,
            "v": self.v,
            "glutamate_release": self.glutamate_release,
        }
