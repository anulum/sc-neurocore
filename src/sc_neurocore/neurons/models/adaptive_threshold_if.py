# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Integrate-and-fire with dynamic threshold. Platkiewicz &

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AdaptiveThresholdIFNeuron:
    """Integrate-and-fire with dynamic threshold. Platkiewicz & Bhatt 2010.

    C dV/dt = -g_L(V - V_rest) + I
    dtheta/dt = -(theta - theta_rest) / tau_theta
    On spike: V -> V_reset, theta += delta_theta

    Reference: Platkiewicz, J. & Brette, R. (2010). J. Neurosci. 30:6891–6902.
    """

    v: float = -65.0
    theta: float = -50.0
    v_rest: float = -65.0
    v_reset: float = -65.0
    theta_rest: float = -50.0
    delta_theta: float = 5.0
    tau_m: float = 10.0
    tau_theta: float = 50.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        for name in ("v", "theta", "v_rest", "v_reset", "theta_rest"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not math.isfinite(self.delta_theta) or self.delta_theta < 0.0:
            raise ValueError("delta_theta must be finite and non-negative")
        if not math.isfinite(self.tau_m) or self.tau_m <= 0.0:
            raise ValueError("tau_m must be finite and positive")
        if not math.isfinite(self.tau_theta) or self.tau_theta <= 0.0:
            raise ValueError("tau_theta must be finite and positive")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self.theta_rest <= self.v_rest:
            raise ValueError("theta_rest must be greater than v_rest")
        if self.theta_rest <= self.v_reset:
            raise ValueError("theta_rest must be greater than v_reset")

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()

        try:
            next_v = self._exact_relaxation(self.v, self.v_rest + current, self.tau_m)
            next_theta = self._exact_relaxation(self.theta, self.theta_rest, self.tau_theta)
        except OverflowError as exc:
            raise ValueError("exact relaxation update must remain finite") from exc
        if not math.isfinite(next_v) or not math.isfinite(next_theta):
            raise ValueError("exact relaxation update must remain finite")

        if next_v >= next_theta:
            spike_theta = next_theta + self.delta_theta
            if not math.isfinite(spike_theta):
                raise ValueError("threshold jump update must remain finite")
            self.v = self.v_reset
            self.theta = spike_theta
            return 1
        self.v = next_v
        self.theta = next_theta
        return 0

    def _exact_relaxation(self, state: float, steady_state: float, tau: float) -> float:
        decay = math.exp(-self.dt / tau)
        candidate = steady_state + (state - steady_state) * decay
        if not math.isfinite(candidate):
            raise ValueError("exact relaxation update must remain finite")
        return candidate

    def _validate_runtime_state(self) -> None:
        if not math.isfinite(self.v):
            raise ValueError("runtime voltage state must be finite")
        if not math.isfinite(self.theta):
            raise ValueError("runtime threshold state must be finite")

    def reset(self) -> None:
        self.v = self.v_rest
        self.theta = self.theta_rest
