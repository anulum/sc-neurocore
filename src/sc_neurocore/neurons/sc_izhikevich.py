# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass
from typing import Dict, Any


from .base import BaseNeuron
from ..utils.rng import RNG


@dataclass
class SCIzhikevichNeuron(BaseNeuron):
    """
    Stochastic Izhikevich neuron (software-only).

    Standard Izhikevich model:
    v' = 0.04*v^2 + 5*v + 140 - u + I + noise
    u' = a*(b*v - u)

    When v >= 30 mV:
    spike, then v <- c, u <- u + d

    Here we add Gaussian noise to v' each step.
    """

    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0
    dt: float = 1.0
    noise_std: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = RNG(self.seed)
        self.reset_state()

    def step(self, input_current: float) -> int:
        # Compute derivatives
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + input_current) * self.dt  # type: ignore
        du = (self.a * (self.b * self.v - self.u)) * self.dt  # type: ignore

        # Add noise to membrane
        if self.noise_std > 0.0:
            dv += float(self._rng.normal(0.0, self.noise_std))

        self.v += dv  # type: ignore
        self.u += du  # type: ignore

        if self.v >= 30.0:  # type: ignore
            # Spike event
            spike = 1
            self.v = self.c
            self.u += self.d  # type: ignore
        else:
            spike = 0
        return spike

    def reset_state(self) -> None:
        self.v = self.c  # membrane potential
        self.u = self.b * self.v  # recovery variable

    def get_state(self) -> Dict[str, Any]:
        return {"v": float(self.v), "u": float(self.u)}
