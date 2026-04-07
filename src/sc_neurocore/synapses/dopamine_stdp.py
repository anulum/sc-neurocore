# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dopamine-Gated STDP Synapse (Izhikevich 2007)

"""Dopamine-gated STDP: learning rate modulated by global reward signal.

Standard STDP with an eligibility trace gated by dopamine concentration.
Weight updates only occur when dopamine is present, solving the distal
reward problem:

    dw/dt  = lr * DA(t) * e(t)
    de/dt  = -e/tau_e + STDP(delta_t) * delta(t_spike)
    dDA/dt = -DA/tau_DA + reward(t)

On a pre-spike: eligibility += a_minus * trace_post (LTD contribution).
On a post-spike: eligibility += a_plus * trace_pre (LTP contribution).

The dopamine signal integrates reward over tau_DA, providing a slow
modulatory signal that gates Hebbian plasticity.

Reference: Izhikevich (2007) "Solving the distal reward problem through
linkage of STDP and dopamine signaling", Cerebral Cortex 17(10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DopamineStdpSynapse:
    """Dopamine-gated STDP synapse (Izhikevich 2007).

    Parameters
    ----------
    weight : float
        Synaptic weight. Default: 0.5.
    w_min : float
        Minimum weight. Default: 0.0.
    w_max : float
        Maximum weight. Default: 1.0.
    tau_e : float
        Eligibility trace time constant (ms). Default: 1000.0.
    tau_da : float
        Dopamine decay time constant (ms). Default: 200.0.
    tau_pre : float
        Pre-synaptic trace time constant (ms). Default: 20.0.
    tau_post : float
        Post-synaptic trace time constant (ms). Default: 20.0.
    a_plus : float
        LTP amplitude. Default: 1.0.
    a_minus : float
        LTD amplitude (negative). Default: -1.0.
    lr : float
        Learning rate. Default: 0.001.
    dt : float
        Integration timestep (ms). Default: 1.0.
    """

    weight: float = 0.5
    w_min: float = 0.0
    w_max: float = 1.0
    tau_e: float = 1000.0
    tau_da: float = 200.0
    tau_pre: float = 20.0
    tau_post: float = 20.0
    a_plus: float = 1.0
    a_minus: float = -1.0
    lr: float = 0.001
    dt: float = 1.0

    eligibility: float = 0.0
    dopamine: float = 0.0
    trace_pre: float = 0.0
    trace_post: float = 0.0

    def step(self, pre_spike: bool, post_spike: bool, reward: float) -> float:
        """Advance one timestep with spike indicators and reward signal.

        Returns the current weight after update.
        """
        # Decay traces.
        self.trace_pre *= math.exp(-self.dt / self.tau_pre)
        self.trace_post *= math.exp(-self.dt / self.tau_post)
        self.eligibility *= math.exp(-self.dt / self.tau_e)
        self.dopamine += (-self.dopamine / self.tau_da + reward) * self.dt

        if pre_spike:
            # LTD from accumulated post-trace.
            self.eligibility += self.a_minus * self.trace_post
            self.trace_pre += 1.0

        if post_spike:
            # LTP from accumulated pre-trace.
            self.eligibility += self.a_plus * self.trace_pre
            self.trace_post += 1.0

        # Dopamine-gated weight update.
        dw = self.lr * self.dopamine * self.eligibility * self.dt
        self.weight = max(self.w_min, min(self.w_max, self.weight + dw))
        return self.weight

    def reset(self) -> None:
        """Reset state to initial conditions."""
        self.eligibility = 0.0
        self.dopamine = 0.0
        self.trace_pre = 0.0
        self.trace_post = 0.0
