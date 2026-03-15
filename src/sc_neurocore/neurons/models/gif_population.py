# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class GIFPopulationNeuron:
    """Mensi et al. 2012 — Generalized IF with escape-rate spiking.

    Stochastic threshold: P(spike|V) = lambda_0 * exp((V - theta) / delta_v).
    Adaptation currents eta decay after each spike.
    """

    v: float = -65.0
    theta: float = -50.0  # mV, baseline threshold
    eta: float = 0.0  # adaptation current, mV
    tau_m: float = 20.0  # ms
    tau_eta: float = 100.0  # ms
    delta_v: float = 2.0  # mV, escape-rate sharpness
    lambda_0: float = 0.001  # base hazard rate, ms^-1
    eta_increment: float = 5.0  # mV, spike-triggered adaptation
    v_rest: float = -65.0
    v_reset: float = -65.0
    dt: float = 0.5
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng()

    def step(self, current: float) -> int:
        # Mensi 2012 Eq. 1-2
        self.v += (-(self.v - self.v_rest) - self.eta + current) / self.tau_m * self.dt
        self.eta *= np.exp(-self.dt / self.tau_eta)
        hazard = self.lambda_0 * np.exp(min((self.v - self.theta) / self.delta_v, 20.0))
        p_spike = 1.0 - np.exp(-hazard * self.dt)
        if self._rng.random() < p_spike:
            self.v = self.v_reset
            self.eta += self.eta_increment
            return 1
        return 0

    def reset(self):
        self.v, self.eta = -65.0, 0.0
