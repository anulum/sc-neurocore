# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Voltage-based STDP (Clopath et al. 2010)

"""Voltage-based STDP that unifies rate and spike-timing plasticity.

Clopath, Büsing, Vasilaki & Gerstner, Nature Neurosci. 13(3):344-352, 2010.

Uses low-pass filtered membrane voltage traces to determine LTP/LTD:
    LTD: dw = -A_LTD * x_bar * H(u_bar_minus - theta_minus)
    LTP: dw =  A_LTP * x * H(u - theta_plus) * H(u_bar_plus - theta_minus)

Where x_bar is a pre-synaptic trace, u_bar_minus/u_bar_plus are slow/fast
voltage traces, and H is the Heaviside step function.

    from sc_neurocore.synapses.clopath_stdp import ClopathSTDP

    syn = ClopathSTDP()
    for t in range(1000):
        syn.step(pre_spike=..., u_post=..., dt=0.1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ClopathSTDP:
    """Voltage-based STDP (Clopath et al. 2010).

    Parameters
    ----------
    a_ltd : float
        LTD amplitude. Default: 14e-5 (Clopath 2010, Table 1).
    a_ltp : float
        LTP amplitude. Default: 8e-5.
    tau_x : float
        Pre-synaptic trace decay (ms). Default: 15.
    tau_minus : float
        Slow voltage trace decay (ms). Default: 10.
    tau_plus : float
        Fast voltage trace decay (ms). Default: 7.
    theta_minus : float
        LTD voltage threshold (mV). Default: -70.6 (rest).
    theta_plus : float
        LTP voltage threshold (mV). Default: -45.3 (depolarization).
    w_min, w_max : float
        Weight bounds.
    """

    a_ltd: float = 14e-5
    a_ltp: float = 8e-5
    tau_x: float = 15.0
    tau_minus: float = 10.0
    tau_plus: float = 7.0
    theta_minus: float = -70.6
    theta_plus: float = -45.3
    w_min: float = 0.0
    w_max: float = 1.0
    weight: float = 0.5

    def __post_init__(self) -> None:
        self.x_bar = 0.0  # low-pass filtered pre-synaptic trace
        self.u_bar_minus = 0.0  # slow voltage trace (LTD)
        self.u_bar_plus = 0.0  # fast voltage trace (LTP)

    def step(self, pre_spike: bool, u_post: float, dt: float = 1.0) -> float:
        """Advance one timestep.

        Parameters
        ----------
        pre_spike : bool
            Whether the pre-synaptic neuron spiked.
        u_post : float
            Post-synaptic membrane voltage (mV).
        dt : float
            Timestep in ms.

        Returns
        -------
        float
            Updated weight.
        """
        decay_x = math.exp(-dt / self.tau_x)
        decay_minus = math.exp(-dt / self.tau_minus)
        decay_plus = math.exp(-dt / self.tau_plus)

        # LTD: pre-synaptic spike × post depolarization (Clopath 2010, Eq. 2)
        if pre_spike:
            ltd = self.a_ltd * self.x_bar * max(0.0, self.u_bar_minus - self.theta_minus)
            self.weight -= ltd

        # LTP: evaluated every timestep, pre contribution via x_bar trace (Clopath 2010, Eq. 3)
        ltp_post = max(0.0, u_post - self.theta_plus)
        ltp_pre = max(0.0, self.u_bar_plus - self.theta_minus)
        if ltp_post > 0 and ltp_pre > 0:
            self.weight += self.a_ltp * self.x_bar * ltp_post * ltp_pre

        self.weight = max(self.w_min, min(self.w_max, self.weight))

        # Update traces: exact exponential filter (no double-decay)
        self.x_bar *= decay_x
        if pre_spike:
            self.x_bar += 1.0
        self.u_bar_minus = decay_minus * self.u_bar_minus + (1 - decay_minus) * u_post
        self.u_bar_plus = decay_plus * self.u_bar_plus + (1 - decay_plus) * u_post

        return self.weight

    def reset(self) -> None:
        self.x_bar = 0.0
        self.u_bar_minus = 0.0
        self.u_bar_plus = 0.0
