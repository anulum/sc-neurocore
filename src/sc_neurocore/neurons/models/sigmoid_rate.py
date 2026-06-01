# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Continuous rate model with sigmoidal transfer. Wilson &

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class SigmoidRateNeuron:
    """Continuous rate model with sigmoidal transfer. Wilson & Cowan 1972 style.

    tau dr/dt = -r + sigma(beta * (input - theta))

    Reference: Wilson, H.R. & Cowan, J.D. (1972). Biophys. J. 12:1–24.
    """

    r: float = 0.0
    tau: float = 10.0
    beta: float = 1.0
    theta: float = 0.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        for field in ("beta", "theta"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if not math.isfinite(self.r):
            raise ValueError("r must be finite")
        if not 0.0 <= self.r <= 1.0:
            raise ValueError("r must be in [0, 1]")
        for field in ("tau", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")

    def step(self, current: float) -> float:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()

        sigma = self._stable_sigmoid(self.beta, current, self.theta)
        next_r = self._exact_relaxation(self.r, sigma)
        self.r = next_r
        return next_r

    def reset(self) -> None:
        self.r = 0.0

    def _validate_runtime_state(self) -> None:
        if not (
            math.isfinite(self.r)
            and math.isfinite(self.beta)
            and math.isfinite(self.theta)
            and math.isfinite(self.tau)
            and math.isfinite(self.dt)
        ):
            raise ValueError("runtime rate state must be finite")
        if not 0.0 <= self.r <= 1.0:
            raise ValueError("runtime rate state must be in [0, 1]")
        if self.tau <= 0.0 or self.dt <= 0.0:
            raise ValueError("runtime time constants must be positive")

    def _exact_relaxation(self, rate: float, steady_state: float) -> float:
        decay = math.exp(-self.dt / self.tau)
        return decay * rate + (1.0 - decay) * steady_state

    @staticmethod
    def _stable_sigmoid(beta: float, current: float, theta: float) -> float:
        delta = current - theta
        z = beta * delta
        if math.isinf(z):
            return 1.0 if z > 0.0 else 0.0
        if not math.isfinite(z):
            raise ValueError("sigmoid argument must remain finite or saturating")
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)
