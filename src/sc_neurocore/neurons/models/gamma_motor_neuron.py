# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Gamma Motor Neuron

from __future__ import annotations

import math
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

    def __post_init__(self) -> None:
        self._validate_state()

    @classmethod
    def static_type(cls) -> GammaMotorNeuron:
        """Static gamma — bag2/chain intrafusal fibres (length-sensitive)."""
        return cls(tau=12.0, tau_adapt=200.0, a_adapt=0.5, dynamic=False)

    def step(self, drive: float = 0.0) -> int:
        self._validate_state()
        if not math.isfinite(drive):
            raise ValueError("drive must be finite")

        v_old = self.v
        adapt_old = self.adapt
        inp = self.gain * max(0.0, drive) - adapt_old
        v_target = self.v_rest + inp
        v_candidate = v_target + (v_old - v_target) * math.exp(-self.dt / self.tau)
        adapt_target = self.a_adapt * (v_candidate - self.v_rest)
        adapt_candidate = adapt_target + (adapt_old - adapt_target) * math.exp(
            -self.dt / self.tau_adapt
        )

        if not math.isfinite(v_candidate) or not math.isfinite(adapt_candidate):
            raise ValueError("gamma motor candidate state must be finite")

        self.v = v_candidate
        self.adapt = adapt_candidate

        if v_candidate >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.adapt = 0.0
        self._validate_state()

    def _validate_state(self) -> None:
        values = (
            self.v,
            self.v_rest,
            self.v_reset,
            self.v_threshold,
            self.tau,
            self.adapt,
            self.tau_adapt,
            self.a_adapt,
            self.gain,
            self.dt,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("gamma motor state and parameters must be finite")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.tau_adapt <= 0.0:
            raise ValueError("tau_adapt must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.gain < 0.0:
            raise ValueError("gain must be non-negative")
        if self.v_reset >= self.v_threshold:
            raise ValueError("v_reset must be below v_threshold")
