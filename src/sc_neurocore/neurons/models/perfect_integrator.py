# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Non-leaky integrate-and-fire. Lapicque 1907 (no leak)

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PerfectIntegratorNeuron:
    """Non-leaky integrate-and-fire. Lapicque 1907 (no leak).

    dV/dt = I / C

    Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §1.3.
    """

    v: float = 0.0
    c_m: float = 1.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        for field in ("v", "v_threshold", "v_reset"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if self.v_threshold <= self.v_reset:
            raise ValueError("v_threshold must be greater than v_reset")
        if self.v >= self.v_threshold:
            raise ValueError("v must be below v_threshold")
        for field in ("c_m", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()
        voltage_increment = current / self.c_m * self.dt
        next_v = self.v + voltage_increment
        if not math.isfinite(voltage_increment) or not math.isfinite(next_v):
            raise ValueError("voltage increment must be finite")
        self.v = next_v
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_reset

    def _validate_runtime_state(self) -> None:
        for field in ("v", "v_threshold", "v_reset", "c_m", "dt"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"runtime {field} must be finite")
        if self.c_m <= 0.0 or self.dt <= 0.0:
            raise ValueError("runtime c_m and dt must be positive")
        if self.v_threshold <= self.v_reset:
            raise ValueError("runtime v_threshold must be greater than v_reset")
        if self.v >= self.v_threshold:
            raise ValueError("runtime v must be below v_threshold before integration")
