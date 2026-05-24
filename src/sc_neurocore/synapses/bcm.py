# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — BCM metaplasticity (Bienenstock-Cooper-Munro 1982)

"""BCM metaplasticity with sliding threshold.

Bienenstock, Cooper & Munro, J. Neurosci. 2(1):32-48, 1982.

The sliding threshold theta_M tracks E[y^2] and determines the
crossover between LTD and LTP:
    dw/dt = eta * y * (y - theta_M) * x
    d(theta_M)/dt = (y^2 - theta_M) / tau_theta

When y > theta_M → LTP; when y < theta_M → LTD.
The threshold slides so high-rate neurons become harder to potentiate.

    from sc_neurocore.synapses.bcm import BCMSynapse

    syn = BCMSynapse(eta=0.01, tau_theta=1000.0)
    for t in range(10000):
        syn.step(pre_rate=0.3, post_rate=0.7, dt=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class BCMSynapse:
    """BCM synapse with sliding modification threshold.

    Parameters
    ----------
    eta : float
        Learning rate.
    tau_theta : float
        Time constant for sliding threshold (ms).
    theta_init : float
        Initial threshold value.
    w_min, w_max : float
        Weight bounds.
    """

    eta: float = 0.01
    tau_theta: float = 1000.0
    theta_init: float = 0.1
    w_min: float = 0.0
    w_max: float = 1.0
    weight: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.eta) or self.eta < 0.0:
            raise ValueError("eta must be finite and non-negative")
        if not math.isfinite(self.tau_theta) or self.tau_theta <= 0.0:
            raise ValueError("tau_theta must be finite and positive")
        if not math.isfinite(self.theta_init) or self.theta_init < 0.0:
            raise ValueError("theta_init must be finite and non-negative")
        for name in ("w_min", "w_max", "weight"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.w_min > self.w_max:
            raise ValueError("w_min must be less than or equal to w_max")
        if not (self.w_min <= self.weight <= self.w_max):
            raise ValueError("weight must be within [w_min, w_max]")

        self.theta_m = self.theta_init

    def step(self, pre_rate: float, post_rate: float, dt: float = 1.0) -> float:
        """Advance one timestep.

        Parameters
        ----------
        pre_rate : float
            Pre-synaptic firing rate (or spike indicator).
        post_rate : float
            Post-synaptic firing rate (or membrane proxy).
        dt : float
            Timestep in ms.

        Returns
        -------
        float
            Updated weight.
        """
        if not math.isfinite(pre_rate) or pre_rate < 0.0:
            raise ValueError("pre_rate must be finite and non-negative")
        if not math.isfinite(post_rate) or post_rate < 0.0:
            raise ValueError("post_rate must be finite and non-negative")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        # BCM update: dw = eta * y * (y - theta_M) * x
        dw = self.eta * post_rate * (post_rate - self.theta_m) * pre_rate * dt
        self.weight += dw
        self.weight = max(self.w_min, min(self.w_max, self.weight))

        # Sliding threshold: d(theta)/dt = (y^2 - theta) / tau_theta
        self.theta_m += (post_rate**2 - self.theta_m) * dt / self.tau_theta

        return self.weight

    def reset(self) -> None:
        self.theta_m = self.theta_init
