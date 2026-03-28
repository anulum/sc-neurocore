# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SRM0 (Spike Response Model, zeroth order)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SRM0Neuron:
    """Spike Response Model, zeroth order (Gerstner & Kistler 2002).

    v(t) = eta(t - t_hat) + integral(kappa(t - s) * I(s) ds)

    Simplified discrete version:
      eta decays after spike: eta(s) = -eta_reset * exp(-s / tau_eta)
      kappa integrates input: v += (I * R - v) * dt / tau_m + eta
      Spike when v > threshold.

    Gerstner, W. & Kistler, W.M. (2002). Spiking Neuron Models.
    Cambridge University Press. Ch. 4.
    """

    v: float = 0.0
    v_rest: float = 0.0
    v_threshold: float = 1.0
    tau_m: float = 20.0
    tau_eta: float = 50.0
    eta_reset: float = 5.0
    resistance: float = 1.0
    dt: float = 1.0

    def __post_init__(self):
        self._eta = 0.0
        self._last_spike_time = -1000.0
        self._t = 0.0

    def step(self, current: float) -> int:
        # Decay refractory kernel
        self._eta *= np.exp(-self.dt / self.tau_eta)

        # Integrate input with eta as effective rest offset
        effective_rest = self.v_rest + self._eta
        dv = (self.resistance * current - (self.v - effective_rest)) * self.dt / self.tau_m
        self.v += dv

        self._t += self.dt

        # Spike detection
        if self.v >= self.v_threshold:
            self.v = self.v_rest
            self._eta = -self.eta_reset
            self._last_spike_time = self._t
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self._eta = 0.0
        self._t = 0.0
        self._last_spike_time = -1000.0

    def get_state(self):
        return {"v": self.v, "eta": self._eta, "t": self._t}
