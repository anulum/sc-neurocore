# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — BrainScaleS-2 — analog AdEx (1000x real-time). Schemmel 2010

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BrainScaleSAdExNeuron:
    """BrainScaleS-2 — analog AdEx (1000x real-time). Schemmel 2010.

    Reference: Schemmel, J. et al. (2010). Proc. ISCAS 2010: 1947–1950.
    """

    v: float = -65.0
    w: float = 0.0
    v_rest: float = -65.0
    v_reset: float = -68.0
    v_threshold: float = -50.0
    delta_t: float = 2.0
    v_rh: float = -55.0
    tau: float = 20.0
    tau_w: float = 100.0
    a: float = 0.5
    b: float = 7.0
    hw_speedup: float = 1000.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        for field in ("v", "w", "v_rest", "v_reset", "v_threshold", "v_rh", "a", "b"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        for field in ("delta_t", "tau", "tau_w", "hw_speedup", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()
        dt_hw = self.dt * self.hw_speedup
        with np.errstate(over="ignore", invalid="ignore"):
            exp_arg = np.clip((self.v - self.v_rh) / self.delta_t, -20.0, 20.0)
            exp_term = self.delta_t * np.exp(exp_arg)
            dt_bio = dt_hw / self.hw_speedup
            dv = (-(self.v - self.v_rest) + exp_term - self.w + current) / self.tau * dt_bio
            dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w * dt_bio
            next_v = self.v + dv
            next_w = self.w + dw
        self._validate_update(next_v, next_w)
        if next_v >= self.v_threshold:
            spike_w = next_w + self.b
            if not math.isfinite(spike_w):
                raise ValueError("spike adaptation update must remain finite")
            self.v = self.v_reset
            self.w = spike_w
            return 1
        self.v = next_v
        self.w = next_w
        return 0

    def _validate_runtime_state(self) -> None:
        if not math.isfinite(self.v):
            raise ValueError("runtime voltage state must be finite")
        if not math.isfinite(self.w):
            raise ValueError("runtime adaptation state must be finite")

    def _validate_update(self, next_v: float, next_w: float) -> None:
        if not math.isfinite(next_v) or not math.isfinite(next_w):
            raise ValueError("BrainScaleS AdEx integrator update must remain finite")

    def reset(self) -> None:
        self.v = self.v_rest
        self.w = 0.0
