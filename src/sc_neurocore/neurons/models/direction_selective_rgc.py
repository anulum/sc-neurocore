# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Direction-Selective Retinal Ganglion Cell

"""Direction-selective retinal ganglion cell (RGC) with temporal derivative.

Implements On- and Off-centre receptive field with centre-surround
antagonism and temporal derivative-based direction selectivity:

    response = w_c * (I - I_prev) +/- w_s * surround_inhibition
    dV/dt = (-V + drive) / tau
    spike if V >= theta

On-centre cells respond to light increase (positive dI/dt),
Off-centre cells respond to light decrease (negative dI/dt).

Reference: Gollisch & Meister (2010) "Eye smarter than scientists believed",
Masland (2012) "The neuronal organization of the retina".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DirectionSelectiveRGC:
    """Direction-selective retinal ganglion cell.

    Parameters
    ----------
    tau : float
        Membrane time constant (ms). Default: 10.0.
    theta : float
        Spike threshold. Default: 0.5.
    is_on_centre : bool
        True for On-centre, False for Off-centre. Default: True.
    w_centre : float
        Centre weight. Default: 1.0.
    w_surround : float
        Surround inhibition weight. Default: 0.3.
    direction_pref : float
        Preferred direction angle (radians). Default: 0.0.
    dt : float
        Integration timestep. Default: 1.0.
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

    @classmethod
    def new_on(cls) -> DirectionSelectiveRGC:
        """Create an On-centre cell."""
        return cls(is_on_centre=True)

    @classmethod
    def new_off(cls) -> DirectionSelectiveRGC:
        """Create an Off-centre cell."""
        return cls(is_on_centre=False)

    def step_rf(self, intensity: float, surround_mean: float) -> int:
        """Step with local intensity and surround mean intensity.

        Returns 1 if spike, 0 otherwise.
        """
        temporal_diff = intensity - self._prev_intensity
        self._prev_intensity = intensity

        if self.is_on_centre:
            centre_response = self.w_centre * temporal_diff
        else:
            centre_response = -self.w_centre * temporal_diff

        self._surround = 0.9 * self._surround + 0.1 * surround_mean
        surround_inhib = self.w_surround * self._surround

        drive = centre_response - surround_inhib
        self.v += (-self.v + drive) / self.tau * self.dt

        if self.v >= self.theta:
            self.v = 0.0
            return 1
        return 0

    def step(self, current: float) -> int:
        """Simple step (no surround)."""
        return self.step_rf(current, 0.0)

    def reset(self) -> None:
        """Reset state to initial conditions."""
        self.v = 0.0
        self._prev_intensity = 0.0
        self._surround = 0.0
