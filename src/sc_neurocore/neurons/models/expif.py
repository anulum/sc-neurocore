# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exponential IF (no adaptation). Fourcaud-Trocmé et al. 2003

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class ExpIFNeuron:
    """Exponential IF (no adaptation). Fourcaud-Trocmé et al. 2003.

    Reference: Fourcaud-Trocmé, N. et al. (2003). J. Neurosci. 23:11628–11640.
    """

    v: float = -65.0
    v_rest: float = -65.0
    v_reset: float = -68.0
    v_threshold: float = -50.0
    v_rh: float = -55.0
    delta_t: float = 2.0
    tau: float = 20.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        for field in ("v", "v_rest", "v_reset", "v_threshold", "v_rh"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        for field in ("delta_t", "tau", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")

    def _rhs(self, v: float, current: float) -> float:
        exp_term = self.delta_t * np.exp(np.clip((v - self.v_rh) / self.delta_t, -20.0, 20.0))
        rhs = (-(v - self.v_rest) + exp_term + current) / self.tau
        if not math.isfinite(rhs):
            raise ValueError("RK4 derivative must remain finite")
        return float(rhs)

    def step(self, current: float) -> int:
        if not math.isfinite(self.v):
            raise ValueError("runtime voltage state must be finite")
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            k1 = self._rhs(self.v, current)
            k2 = self._rhs(self.v + 0.5 * self.dt * k1, current)
            k3 = self._rhs(self.v + 0.5 * self.dt * k2, current)
            k4 = self._rhs(self.v + self.dt * k3, current)
            next_v = self.v + self.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if not math.isfinite(next_v):
            raise ValueError("RK4 update must remain finite")

        self.v = next_v

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
