# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic Izhikevich neuron (software-only)

from __future__ import annotations
from typing import Any
from dataclasses import dataclass
from typing import Dict


from .base import BaseNeuron
from ..utils.rng import RNG
from ..constants import (
    IZH_A,
    IZH_B,
    IZH_C,
    IZH_D,
    IZH_SPIKE_THRESHOLD,
    LIF_DT,
)


@dataclass
class SCIzhikevichNeuron(BaseNeuron):
    """
    Stochastic Izhikevich neuron (software-only).

    Standard Izhikevich model (IEEE TNN 14(6), 2003):
    v' = 0.04*v^2 + 5*v + 140 - u + I + noise
    u' = a*(b*v - u)

    When v >= 30 mV: spike, then v <- c, u <- u + d.

    Example
    -------
    >>> neuron = SCIzhikevichNeuron(noise_std=0.0)
    >>> spikes = [neuron.step(10.0) for _ in range(100)]
    >>> sum(spikes) > 0  # regular spiking with I=10
    True
    """

    a: float = IZH_A
    b: float = IZH_B
    c: float = IZH_C
    d: float = IZH_D
    dt: float = LIF_DT
    noise_std: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = RNG(self.seed)
        self.v: float = self.c
        self.u: float = self.b * self.c
        self.reset_state()

    def step(self, input_current: float) -> int:
        # Two half-steps for numerical stability on 0.04v² term.
        # Izhikevich (2003) recommends dt ≤ 0.5 ms; we split each dt into two.
        half_dt = self.dt * 0.5
        for _ in range(2):
            dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + input_current) * half_dt
            du = (self.a * (self.b * self.v - self.u)) * half_dt
            self.v += dv
            self.u += du

        if self.noise_std > 0.0:
            self.v += float(self._rng.normal(0.0, self.noise_std))

        if self.v >= IZH_SPIKE_THRESHOLD:
            spike = 1
            self.v = self.c
            self.u += self.d
        else:
            spike = 0
        return spike

    def reset_state(self) -> None:
        self.v = self.c  # membrane potential
        self.u = self.b * self.v  # recovery variable

    def get_state(self) -> Dict[str, Any]:
        return {"v": float(self.v), "u": float(self.u)}
