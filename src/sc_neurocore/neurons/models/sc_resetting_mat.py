# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — project resetting MAT modification

from __future__ import annotations

import math
from dataclasses import dataclass

_VOLTAGE_MIN = -200.0
_VOLTAGE_MAX = 100.0
_THETA_MAX = 1.0e9


@dataclass
class SCResettingMATNeuron:
    """SC candidate-first RK4 adaptive-threshold neuron with voltage reset.

    This project model preserves the historical SC-NeuroCore ``MATNeuron``
    recurrence under an explicit identity. It is not attributed to the
    non-resetting MAT* equations of Kobayashi et al. (2009).
    """

    v: float = -70.0
    theta1: float = 0.0
    theta2: float = 0.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold_base: float = -50.0
    tau_m: float = 10.0
    tau_1: float = 10.0
    tau_2: float = 200.0
    h1: float = 5.0
    h2: float = 3.0
    resistance: float = 1.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        """Validate the numerical contract before the first integration step."""
        self._validate_state()

    def _validate_state(self) -> None:
        """Reject invalid SC resetting-MAT parameters or state before mutation."""
        finite_values = (
            self.v,
            self.theta1,
            self.theta2,
            self.v_rest,
            self.v_reset,
            self.v_threshold_base,
            self.tau_m,
            self.tau_1,
            self.tau_2,
            self.h1,
            self.h2,
            self.resistance,
            self.dt,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("SCResettingMATNeuron state and parameters must be finite")
        if not (_VOLTAGE_MIN <= self.v <= _VOLTAGE_MAX):
            raise ValueError("SCResettingMATNeuron voltage is outside the safety envelope")
        if not (_VOLTAGE_MIN <= self.v_reset <= _VOLTAGE_MAX):
            raise ValueError("SCResettingMATNeuron reset voltage is outside the safety envelope")
        if not (0.0 <= self.theta1 <= _THETA_MAX and 0.0 <= self.theta2 <= _THETA_MAX):
            raise ValueError(
                "SCResettingMATNeuron threshold adaptation is outside the safety envelope"
            )
        if not (0.0 <= self.h1 <= _THETA_MAX and 0.0 <= self.h2 <= _THETA_MAX):
            raise ValueError(
                "SCResettingMATNeuron threshold increments are outside the safety envelope"
            )
        if self.tau_m <= 0.0 or self.tau_1 <= 0.0 or self.tau_2 <= 0.0:
            raise ValueError("SCResettingMATNeuron time constants must be positive")
        if self.resistance <= 0.0 or self.dt <= 0.0:
            raise ValueError("SCResettingMATNeuron resistance and timestep must be positive")

    def _derivatives(
        self, v: float, theta1: float, theta2: float, current: float
    ) -> tuple[float, float, float]:
        """Return the SC membrane and threshold right-hand side."""
        dv = (-(v - self.v_rest) + self.resistance * current) / self.tau_m
        return dv, -theta1 / self.tau_1, -theta2 / self.tau_2

    def _rk4_candidate(self, current: float) -> tuple[float, float, float]:
        """Advance ``(v, theta1, theta2)`` with one candidate-first RK4 step."""
        k1v, k1t1, k1t2 = self._derivatives(self.v, self.theta1, self.theta2, current)
        k2v, k2t1, k2t2 = self._derivatives(
            self.v + 0.5 * self.dt * k1v,
            self.theta1 + 0.5 * self.dt * k1t1,
            self.theta2 + 0.5 * self.dt * k1t2,
            current,
        )
        k3v, k3t1, k3t2 = self._derivatives(
            self.v + 0.5 * self.dt * k2v,
            self.theta1 + 0.5 * self.dt * k2t1,
            self.theta2 + 0.5 * self.dt * k2t2,
            current,
        )
        k4v, k4t1, k4t2 = self._derivatives(
            self.v + self.dt * k3v,
            self.theta1 + self.dt * k3t1,
            self.theta2 + self.dt * k3t2,
            current,
        )
        scale = self.dt / 6.0
        return (
            self.v + scale * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
            self.theta1 + scale * (k1t1 + 2.0 * k2t1 + 2.0 * k3t1 + k4t1),
            self.theta2 + scale * (k1t2 + 2.0 * k2t2 + 2.0 * k3t2 + k4t2),
        )

    def step(self, current: float) -> int:
        """Advance one SC resetting-MAT step and return ``1`` on a spike."""
        if not math.isfinite(current):
            raise ValueError("SCResettingMATNeuron input current must be finite")
        self._validate_state()
        v_candidate, theta1_candidate, theta2_candidate = self._rk4_candidate(current)
        if not (
            math.isfinite(v_candidate)
            and math.isfinite(theta1_candidate)
            and math.isfinite(theta2_candidate)
            and _VOLTAGE_MIN <= v_candidate <= _VOLTAGE_MAX
            and 0.0 <= theta1_candidate <= _THETA_MAX
            and 0.0 <= theta2_candidate <= _THETA_MAX
        ):
            raise ValueError("SCResettingMATNeuron RK4 candidate left the safety envelope")
        if v_candidate >= self.v_threshold_base + theta1_candidate + theta2_candidate:
            theta1_after_spike = theta1_candidate + self.h1
            theta2_after_spike = theta2_candidate + self.h2
            if theta1_after_spike > _THETA_MAX or theta2_after_spike > _THETA_MAX:
                raise ValueError(
                    "SCResettingMATNeuron post-spike adaptation left the safety envelope"
                )
            self.v = self.v_reset
            self.theta1 = theta1_after_spike
            self.theta2 = theta2_after_spike
            return 1
        self.v = v_candidate
        self.theta1 = theta1_candidate
        self.theta2 = theta2_candidate
        return 0

    def reset(self) -> None:
        """Restore voltage and adaptive thresholds to the SC resting state."""
        self.v = self.v_rest
        self.theta1 = 0.0
        self.theta2 = 0.0
