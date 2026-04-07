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

    def p_open(self, displacement: float) -> float:
        """Boltzmann activation of MET channels."""
        return 1.0 / (1.0 + math.exp(-(displacement - self.x0) / self.delta))

    def step(self, displacement: float) -> int:
        """Step with basilar membrane displacement.

        Returns 1 if glutamate_release > 0.5, else 0.
        """
        po = self.p_open(displacement)
        i_met = self.g_max * po * (self.v - self.e_met)
        dv = (-self.g_l * (self.v - self.e_l) - i_met) / self.cap
        self.v += dv * self.dt

        # Graded glutamate release proportional to depolarisation.
        self.glutamate_release = max((self.v + 60.0), 0.0) / 40.0
        return 1 if self.glutamate_release > 0.5 else 0

    def reset(self) -> None:
        """Reset state to resting potential."""
        self.v = self.e_l
        self.glutamate_release = 0.0
