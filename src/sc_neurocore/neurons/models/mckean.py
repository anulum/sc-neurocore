# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-bound space-clamped McKean Heaviside system

"""Source-faithful space-clamped McKean neuron."""

from __future__ import annotations

import math
from dataclasses import dataclass

_STATE_BOUND = 1.0e6


@dataclass
class McKeanNeuron:
    """McKean's discontinuous FitzHugh-Nagumo caricature.

    The equations are the source-bound space-clamped system of Tonnelier
    (2002), equations (1.3)-(1.6), following McKean (1970):
    ``dv/dt=-lambda*v+mu*H(v-a)-w+I`` and ``dw/dt=b*v``.
    The numerical specialization declares ``H(0)=1`` and samples an event on
    upward crossing of the switching line; the ODE has no spike reset.
    """

    v: float = 0.0
    w: float = 0.0
    a: float = 0.25
    lambda_: float = 1.0
    mu: float = 1.0
    b: float = 0.01
    dt: float = 0.1

    def __post_init__(self) -> None:
        """Validate source state and parameter constraints."""
        self._validate_state()

    def _validate_state(self) -> None:
        for field in ("v", "w", "a", "lambda_", "mu", "b", "dt"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field} must be a real scalar")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
            setattr(self, field, value)
        if abs(self.v) > _STATE_BOUND or abs(self.w) > _STATE_BOUND:
            raise ValueError("McKean state is outside the safety envelope")
        if self.a <= 0.0 or self.lambda_ <= 0.0 or self.mu <= 0.0 or self.b <= 0.0:
            raise ValueError("a, lambda_, mu, and b must be positive")
        if self.mu <= self.lambda_ * self.a:
            raise ValueError("source constraint mu > lambda_ * a is required")
        if self.dt <= 0.0 or self.dt > 1.0:
            raise ValueError("dt must be in the enrolled interval (0, 1]")

    def _derivatives(self, v: float, w: float, current: float) -> tuple[float, float]:
        """Evaluate the right-continuous Heaviside source equations."""
        heaviside = 1.0 if v >= self.a else 0.0
        return -self.lambda_ * v + self.mu * heaviside - w + current, self.b * v

    def _rk4_candidate(self, current: float) -> tuple[float, float]:
        dt = self.dt
        k1_v, k1_w = self._derivatives(self.v, self.w, current)
        k2_v, k2_w = self._derivatives(self.v + 0.5 * dt * k1_v, self.w + 0.5 * dt * k1_w, current)
        k3_v, k3_w = self._derivatives(self.v + 0.5 * dt * k2_v, self.w + 0.5 * dt * k2_w, current)
        k4_v, k4_w = self._derivatives(self.v + dt * k3_v, self.w + dt * k3_w, current)
        scale = dt / 6.0
        return (
            self.v + scale * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v),
            self.w + scale * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w),
        )

    def step(self, current: float = 0.0) -> int:
        """Advance one RK4 sample atomically and report a switching-line crossing."""
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            raise TypeError("current must be a real scalar")
        current = float(current)
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_state()
        v_candidate, w_candidate = self._rk4_candidate(current)
        if not (
            math.isfinite(v_candidate)
            and math.isfinite(w_candidate)
            and abs(v_candidate) <= _STATE_BOUND
            and abs(w_candidate) <= _STATE_BOUND
        ):
            raise ValueError("McKean RK4 candidate left the safety envelope")
        event = int(self.v < self.a <= v_candidate)
        self.v, self.w = v_candidate, w_candidate
        return event

    def reset(self) -> None:
        """Restore the source equilibrium state."""
        self.v = 0.0
        self.w = 0.0
