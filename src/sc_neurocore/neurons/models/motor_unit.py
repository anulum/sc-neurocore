# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Motor Unit (Alpha Motor Neuron + Muscle Fibre)

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MotorUnit:
    """Motor unit — alpha motor neuron + muscle fibre.

    Each spike triggers a muscle twitch. Force output is summation of
    overlapping twitches (rate coding). Twitch modelled as critically-
    damped second-order: f(t) = A · (t/τ) · exp(1 - t/τ).

    Reference: Fuglevand et al. (1993) J Neurophysiol 70(6);
    Heckman & Enoka (2012) Compr Physiol 2(4).
    """

    v: float = -65.0
    v_rest: float = -65.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 10.0
    adapt: float = 0.0
    tau_adapt: float = 100.0
    a_adapt: float = 0.2
    gain: float = 1.0
    force: float = 0.0
    twitch_amp: float = 0.05
    tau_twitch: float = 90.0
    force_decay: float = 0.0
    dt: float = 0.5

    @classmethod
    def slow(cls) -> MotorUnit:
        """Slow motor unit (type S): small, fatigue-resistant, low force."""
        return cls()

    @classmethod
    def fast(cls) -> MotorUnit:
        """Fast motor unit (type FF): large, fatigable, high force."""
        return cls(
            tau_m=6.0,
            tau_adapt=50.0,
            a_adapt=0.1,
            twitch_amp=0.3,
            tau_twitch=30.0,
        )

    def step(self, drive: float = 0.0) -> int:
        inp = self.gain * max(0.0, drive) - self.adapt
        self.v += (-(self.v - self.v_rest) + inp) / self.tau_m * self.dt
        self.adapt += (
            (self.a_adapt * (self.v - self.v_rest) - self.adapt) / self.tau_adapt * self.dt
        )

        self.force *= math.exp(-self.dt / self.tau_twitch)

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.force = min(1.0, self.force + self.twitch_amp)
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.adapt = 0.0
        self.force = 0.0
