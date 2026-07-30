# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Kobayashi 2009 MAT(1) non-resetting neuron

from __future__ import annotations

import math
from dataclasses import dataclass

_VOLTAGE_MIN = -200.0
_VOLTAGE_MAX = 200.0
_THRESHOLD_HISTORY_MAX = 1.0e9


@dataclass
class NonResettingLIFNeuron:
    """Source-faithful one-timescale MAT(1) non-resetting LIF neuron.

    The membrane follows equation 1 of Kobayashi, Tsubo, and Shinomoto
    (2009). ``theta`` stores the single exponentially decaying spike-history
    contribution from equations 2-3; the instantaneous threshold is
    ``omega + theta``. A spike raises that history and starts the paper's
    2 ms absolute refractory interval, but never resets ``v``.

    The paper identifies 50 ms as the optimal MAT(1) threshold timescale but
    fits threshold amplitude and baseline per neuron. The defaults therefore
    form a documented numerical specialization, not a universal cell fit.

    Reference: R. Kobayashi, Y. Tsubo, and S. Shinomoto, Frontiers in
    Computational Neuroscience 3:9 (2009), doi:10.3389/neuro.10.009.2009.
    """

    v: float = 0.0
    theta: float = 0.0
    refractory_remaining: float = 0.0
    omega: float = 19.0
    tau_m: float = 5.0
    tau_theta: float = 50.0
    alpha: float = 37.0
    resistance: float = 50.0
    refractory_period: float = 2.0
    dt: float = 0.001

    def __post_init__(self) -> None:
        """Validate the complete MAT(1) state and parameter contract."""
        self._validate_state()

    @property
    def threshold(self) -> float:
        """Return the instantaneous adaptive threshold in millivolts."""
        return self.omega + self.theta

    def _validate_state(self) -> None:
        """Reject invalid state and parameters before any mutation."""
        values = (
            self.v,
            self.theta,
            self.refractory_remaining,
            self.omega,
            self.tau_m,
            self.tau_theta,
            self.alpha,
            self.resistance,
            self.refractory_period,
            self.dt,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("NonResettingLIFNeuron state and parameters must be finite")
        if not (_VOLTAGE_MIN <= self.v <= _VOLTAGE_MAX):
            raise ValueError("NonResettingLIFNeuron voltage is outside the safety envelope")
        if not (0.0 <= self.theta <= _THRESHOLD_HISTORY_MAX):
            raise ValueError(
                "NonResettingLIFNeuron threshold history is outside the safety envelope"
            )
        if not (-_THRESHOLD_HISTORY_MAX <= self.omega <= _THRESHOLD_HISTORY_MAX):
            raise ValueError(
                "NonResettingLIFNeuron baseline threshold is outside the safety envelope"
            )
        if not (0.0 <= self.alpha <= _THRESHOLD_HISTORY_MAX):
            raise ValueError(
                "NonResettingLIFNeuron threshold increment is outside the safety envelope"
            )
        if self.tau_m <= 0.0 or self.tau_theta <= 0.0:
            raise ValueError("NonResettingLIFNeuron time constants must be positive")
        if self.resistance <= 0.0 or self.dt <= 0.0 or self.refractory_period < 0.0:
            raise ValueError(
                "NonResettingLIFNeuron resistance/timestep must be positive and refractory nonnegative"
            )
        if not (0.0 <= self.refractory_remaining <= self.refractory_period):
            raise ValueError("NonResettingLIFNeuron refractory state is outside its interval")

    def step(self, current: float) -> int:
        """Advance one source MAT(1) sample and return ``1`` on a spike.

        Voltage continues evolving during the refractory interval. Candidate
        state is validated before commit, so invalid input or overflow leaves
        the complete neuron state unchanged.
        """
        if not math.isfinite(current):
            raise ValueError("NonResettingLIFNeuron input current must be finite")
        self._validate_state()

        v_candidate = self.v + self.dt * (-self.v + self.resistance * current) / self.tau_m
        theta_candidate = self.theta * math.exp(-self.dt / self.tau_theta)
        refractory_candidate = max(0.0, self.refractory_remaining - self.dt)
        if not all(
            math.isfinite(value) for value in (v_candidate, theta_candidate, refractory_candidate)
        ):
            raise ValueError("NonResettingLIFNeuron candidate state must be finite")
        if not (_VOLTAGE_MIN <= v_candidate <= _VOLTAGE_MAX):
            raise ValueError("NonResettingLIFNeuron voltage candidate left the safety envelope")
        if not (0.0 <= theta_candidate <= _THRESHOLD_HISTORY_MAX):
            raise ValueError("NonResettingLIFNeuron threshold candidate left the safety envelope")

        spike = refractory_candidate == 0.0 and v_candidate >= self.omega + theta_candidate
        if spike:
            theta_candidate += self.alpha
            if not math.isfinite(theta_candidate) or theta_candidate > _THRESHOLD_HISTORY_MAX:
                raise ValueError(
                    "NonResettingLIFNeuron post-spike threshold left the safety envelope"
                )
            refractory_candidate = self.refractory_period

        self.v = v_candidate
        self.theta = theta_candidate
        self.refractory_remaining = refractory_candidate
        return int(spike)

    def reset(self) -> None:
        """Restore zero-rest voltage, threshold history, and refractory state."""
        self.v = 0.0
        self.theta = 0.0
        self.refractory_remaining = 0.0
