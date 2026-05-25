# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lapicque 1907 — classical RC integrate-and-fire

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class LapicqueNeuron:
    """Lapicque 1907 — classical RC integrate-and-fire.

    tau * dv/dt = -(v - v_rest) + R * I

    Reference: Lapicque, L. (1907). J. Physiol. Pathol. Gén. 9:620–635.
    """

    v: float = 0.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_threshold: float = 1.0
    tau: float = 20.0
    resistance: float = 1.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        self._validate_runtime_state()

    def _validate_runtime_state(self) -> None:
        for field in ("v", "v_rest", "v_reset", "v_threshold"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if self.v_threshold <= self.v_rest:
            raise ValueError("v_threshold must be greater than v_rest")
        if self.v_threshold <= self.v_reset:
            raise ValueError("v_threshold must be greater than v_reset")
        if self.v >= self.v_threshold:
            raise ValueError("v must be below v_threshold")
        for field in ("tau", "resistance", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()
        dv = (-(self.v - self.v_rest) + self.resistance * current) / self.tau * self.dt
        next_v = self.v + dv
        if not math.isfinite(dv) or not math.isfinite(next_v):
            raise ValueError("voltage increment must be finite")
        self.v = next_v

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
