# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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

    def __post_init__(self):
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
        # BCM update: dw = eta * y * (y - theta_M) * x
        dw = self.eta * post_rate * (post_rate - self.theta_m) * pre_rate * dt
        self.weight += dw
        self.weight = max(self.w_min, min(self.w_max, self.weight))

        # Sliding threshold: d(theta)/dt = (y^2 - theta) / tau_theta
        self.theta_m += (post_rate ** 2 - self.theta_m) * dt / self.tau_theta

        return self.weight

    def reset(self):
        self.theta_m = self.theta_init
