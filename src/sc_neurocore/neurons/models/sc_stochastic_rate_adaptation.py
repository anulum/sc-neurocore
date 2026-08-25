# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — project stochastic rate-adaptation recurrence

"""SC project stochastic rate-adaptation model preserved from BendaHerzNeuron."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SCStochasticRateAdaptationNeuron:
    """SC logistic rate adaptation with exponential-hazard spike sampling.

    This count-neutral project model preserves the former ``BendaHerzNeuron``
    behavior. It is not attributed to the deterministic Benda–Herz phase
    generator.
    """

    a: float = 0.0
    f_max: float = 200.0
    beta: float = 0.1
    i_half: float = 5.0
    tau_a: float = 100.0
    delta_a: float = 0.5
    dt: float = 1.0
    seed: int | None = None
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.a) or self.a < 0.0:
            raise ValueError("a must be finite and non-negative")
        for name in ("f_max", "beta", "tau_a", "dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.i_half):
            raise ValueError("i_half must be finite")
        if not math.isfinite(self.delta_a) or self.delta_a < 0.0:
            raise ValueError("delta_a must be finite and non-negative")
        if self.seed is not None:
            if (
                isinstance(self.seed, bool)
                or not isinstance(self.seed, (int, np.integer))
                or self.seed < 0
                or self.seed > 2**64 - 1
            ):
                raise ValueError("seed must be None or a uint64-compatible integer")
        self._rng = np.random.default_rng(self.seed)

    def _f_onset(self, x: float) -> float:
        z = self.beta * (x - self.i_half)
        if z == math.inf:
            return self.f_max
        if z == -math.inf:
            return 0.0
        if not math.isfinite(z):
            raise ValueError("onset rate argument must be finite")
        if z >= 0.0:
            return self.f_max / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return self.f_max * exp_z / (1.0 + exp_z)

    def _adaptation_rhs(self, a: float, current: float) -> tuple[float, float]:
        if not math.isfinite(a) or a < 0.0:
            raise ValueError("adaptation RK4 stage must be finite and non-negative")
        rate = self._f_onset(current - a)
        if not math.isfinite(rate) or rate < 0.0 or rate > self.f_max:
            raise ValueError("onset rate must be finite and bounded")
        return -a / self.tau_a + self.delta_a * rate, rate

    def _rk4_candidate(self, current: float) -> tuple[float, float]:
        k1, r1 = self._adaptation_rhs(self.a, current)
        k2, r2 = self._adaptation_rhs(self.a + 0.5 * self.dt * k1, current)
        k3, r3 = self._adaptation_rhs(self.a + 0.5 * self.dt * k2, current)
        k4, r4 = self._adaptation_rhs(self.a + self.dt * k3, current)
        next_a = self.a + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        average_rate = (r1 + 2.0 * r2 + 2.0 * r3 + r4) / 6.0
        hazard = average_rate * self.dt / 1000.0
        if not math.isfinite(hazard) or hazard < 0.0:
            raise ValueError("spike hazard must be finite and non-negative")
        probability = -math.expm1(-hazard)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("spike probability must be finite and within [0, 1]")
        if not math.isfinite(next_a) or next_a < 0.0:
            raise ValueError("adaptation RK4 candidate must be finite and non-negative")
        return next_a, probability

    def step_with_uniform(self, current: float, uniform: float) -> int:
        """Advance using an explicit uniform variate for backend parity."""
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        if not math.isfinite(uniform) or not 0.0 <= uniform < 1.0:
            raise ValueError("uniform must be finite and within [0, 1)")
        next_a, probability = self._rk4_candidate(current)
        self.a = next_a
        return int(uniform < probability)

    def step(self, current: float) -> int:
        return self.step_with_uniform(current, float(self._rng.random()))

    def reset(self) -> None:
        self.a = 0.0
