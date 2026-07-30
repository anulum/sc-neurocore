# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained project non-resetting adaptive LIF recurrence

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SCNonResettingAdaptiveLIFNeuron:
    """Retained SC exact-relaxation adaptive LIF with no voltage reset.

    This is the historical project recurrence formerly exposed as
    ``NonResettingLIFNeuron``. Both voltage and threshold use exact affine
    relaxation over one configured sample. A level event raises threshold by
    ``delta_theta``; voltage is never reset and no refractory gate is applied.

    The recurrence is project-defined and intentionally carries no external
    paper attribution. Use :class:`NonResettingLIFNeuron` for source MAT(1).
    """

    v: float = -65.0
    theta: float = -50.0
    v_rest: float = -65.0
    theta_rest: float = -50.0
    delta_theta: float = 5.0
    tau_m: float = 10.0
    tau_theta: float = 50.0
    r_m: float = 1.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        """Validate the complete project recurrence before first use."""
        self._validate_runtime_state()

    def step(self, current: float) -> int:
        """Advance one exact-relaxation sample with atomic failure."""
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()
        membrane_steady_state = self.v_rest + self.r_m * current
        if not math.isfinite(membrane_steady_state):
            raise ValueError("membrane exact relaxation update must remain finite")
        next_v = self._exact_relaxation(self.v, membrane_steady_state, self.tau_m)
        next_theta = self._exact_relaxation(self.theta, self.theta_rest, self.tau_theta)
        if not math.isfinite(next_v):
            raise ValueError("membrane exact relaxation update must remain finite")
        if not math.isfinite(next_theta):
            raise ValueError("threshold exact relaxation update must remain finite")

        spike = next_v >= next_theta
        if spike:
            next_theta += self.delta_theta
            if not math.isfinite(next_theta):
                raise ValueError("threshold exact relaxation update must remain finite")

        self.v = next_v
        self.theta = next_theta
        return int(spike)

    def reset(self) -> None:
        """Restore voltage and adaptive threshold to their configured rests."""
        self.v = self.v_rest
        self.theta = self.theta_rest

    def _exact_relaxation(self, state: float, steady_state: float, tau: float) -> float:
        """Return the exact affine relaxation over one configured sample."""
        decay = math.exp(-self.dt / tau)
        return decay * state + (1.0 - decay) * steady_state

    def _validate_runtime_state(self) -> None:
        """Reject non-finite or physically invalid runtime state."""
        for field in (
            "v",
            "theta",
            "v_rest",
            "theta_rest",
            "delta_theta",
            "tau_m",
            "tau_theta",
            "r_m",
            "dt",
        ):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"runtime {field} must be finite")
        if self.delta_theta < 0.0 or self.r_m < 0.0:
            raise ValueError("runtime delta_theta and r_m must be non-negative")
        if self.tau_m <= 0.0 or self.tau_theta <= 0.0 or self.dt <= 0.0:
            raise ValueError("runtime tau_m, tau_theta, and dt must be positive")
