# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Triplet STDP (Pfister & Gerstner 2006)

"""Triplet STDP: extends pair-based STDP with triplet interactions.

Pfister & Gerstner, J. Neurosci. 26(38):9673-9682, 2006.

The triplet rule uses four traces (r1, r2, o1, o2) to capture
pre-post-post and pre-pre-post interactions. This explains:
- Rate-dependence of LTP/LTD
- The BCM crossover frequency
- Frequency-dependent pairing protocols

    from sc_neurocore.synapses.triplet_stdp import TripletSTDP

    synapse = TripletSTDP(tau_plus=16.8, tau_minus=33.7,
                          tau_x=101.0, tau_y=125.0)
    for t in range(1000):
        synapse.step(pre_spike=..., post_spike=..., dt=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TripletSTDP:
    """Triplet STDP synapse (Pfister-Gerstner 2006).

    Parameters
    ----------
    tau_plus : float
        Pre-synaptic trace decay (ms). Default: 16.8 (visual cortex fit).
    tau_minus : float
        Post-synaptic trace decay (ms). Default: 33.7.
    tau_x : float
        Slow pre-synaptic trace decay (ms). Default: 101.
    tau_y : float
        Slow post-synaptic trace decay (ms). Default: 125.
    a2_plus : float
        Pair LTP amplitude. Default: 7.5e-10.
    a3_plus : float
        Triplet LTP amplitude. Default: 9.3e-3.
    a2_minus : float
        Pair LTD amplitude. Default: 7.0e-3.
    a3_minus : float
        Triplet LTD amplitude. Default: 2.3e-4.
    w_min : float
        Minimum weight. Default: 0.0.
    w_max : float
        Maximum weight. Default: 1.0.
    """

    tau_plus: float = 16.8
    tau_minus: float = 33.7
    tau_x: float = 101.0
    tau_y: float = 125.0
    a2_plus: float = 7.5e-10
    a3_plus: float = 9.3e-3
    a2_minus: float = 7.0e-3
    a3_minus: float = 2.3e-4
    w_min: float = 0.0
    w_max: float = 1.0
    weight: float = 0.5

    def __post_init__(self) -> None:
        self.r1 = 0.0  # fast pre-synaptic trace
        self.r2 = 0.0  # slow pre-synaptic trace
        self.o1 = 0.0  # fast post-synaptic trace
        self.o2 = 0.0  # slow post-synaptic trace

    def step(self, pre_spike: bool, post_spike: bool, dt: float = 1.0) -> float:
        """Advance one timestep.

        Returns the current weight after update.
        """
        import math

        # Decay traces
        self.r1 *= math.exp(-dt / self.tau_plus)
        self.r2 *= math.exp(-dt / self.tau_x)
        self.o1 *= math.exp(-dt / self.tau_minus)
        self.o2 *= math.exp(-dt / self.tau_y)

        # Weight updates on spikes
        if post_spike:
            # LTP: pair + triplet pre-post-post
            self.weight += self.r1 * (self.a2_plus + self.a3_plus * self.o2)
        if pre_spike:
            # LTD: pair + triplet pre-pre-post
            self.weight -= self.o1 * (self.a2_minus + self.a3_minus * self.r2)

        # Clamp
        self.weight = max(self.w_min, min(self.w_max, self.weight))

        # Update traces after weight change (order matters — Pfister 2006 Eq. 3-4)
        if pre_spike:
            self.r1 += 1.0
            self.r2 += 1.0
        if post_spike:
            self.o1 += 1.0
            self.o2 += 1.0

        return self.weight

    def reset(self) -> None:
        self.r1 = self.r2 = self.o1 = self.o2 = 0.0
