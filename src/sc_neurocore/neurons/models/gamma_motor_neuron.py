# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Gamma Motor Neuron

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GammaMotorNeuron:
    """Gamma motor neuron — innervates intrafusal fibres of muscle spindles.

    Simple LIF with spike-frequency adaptation. Two subtypes: dynamic
    (bag1, velocity-sensitive) and static (bag2/chain, length-sensitive).

    Reference: Prochazka & Hulliger (1989) Prog Brain Res 80;
    Taylor et al. (1999) J Physiol 519(3).
    """

    v: float = -65.0
    v_rest: float = -65.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau: float = 8.0
    adapt: float = 0.0
    tau_adapt: float = 100.0
    a_adapt: float = 0.3
    gain: float = 1.0
    dynamic: bool = True
    dt: float = 0.5

    @classmethod
    def static_type(cls) -> GammaMotorNeuron:
        """Static gamma — bag2/chain intrafusal fibres (length-sensitive)."""
        return cls(tau=12.0, tau_adapt=200.0, a_adapt=0.5, dynamic=False)

    def step(self, drive: float = 0.0) -> int:
        inp = self.gain * max(0.0, drive) - self.adapt
        self.v += (-(self.v - self.v_rest) + inp) / self.tau * self.dt
        self.adapt += (
            self.a_adapt * (self.v - self.v_rest) - self.adapt
        ) / self.tau_adapt * self.dt

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.adapt = 0.0
