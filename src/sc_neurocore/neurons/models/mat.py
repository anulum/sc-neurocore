# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Kobayashi 2009 MAT* adaptive-threshold neuron

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Self

_VOLTAGE_MIN = -200.0
_VOLTAGE_MAX = 200.0
_THETA_MAX = 1.0e9


@dataclass
class MATNeuron:
    """Non-resetting MAT* neuron from Kobayashi, Tsubo, and Shinomoto (2009).

    ``v`` is measured relative to the resting potential. The membrane follows
    forward Euler, while the two spike-history terms use their exact
    exponential decay. A spike raises the adaptive threshold but never resets
    the membrane voltage. The default parameter set is the paper's regular-
    spiking (RS) example, not a universal cortical-cell calibration.

    Reference: R. Kobayashi, Y. Tsubo, and S. Shinomoto, Frontiers in
    Computational Neuroscience 3:9 (2009), doi:10.3389/neuro.10.009.2009.
    """

    v: float = 0.0
    theta1: float = 0.0
    theta2: float = 0.0
    refractory_remaining: float = 0.0
    omega: float = 19.0
    tau_m: float = 5.0
    tau_1: float = 10.0
    tau_2: float = 200.0
    alpha_1: float = 37.0
    alpha_2: float = 2.0
    resistance: float = 50.0
    refractory_period: float = 2.0
    dt: float = 0.001

    def __post_init__(self) -> None:
        """Validate the complete source-model contract before first use."""
        self._validate_state()

    @classmethod
    def regular_spiking(cls, **overrides: float) -> Self:
        """Construct the paper's regular-spiking example profile."""
        return cls(**overrides)

    @classmethod
    def intrinsically_bursting(cls, **overrides: float) -> Self:
        """Construct the paper's intrinsically-bursting example profile."""
        return cls(omega=26.0, alpha_1=1.7, alpha_2=2.0, **overrides)

    @classmethod
    def fast_spiking(cls, **overrides: float) -> Self:
        """Construct the paper's fast-spiking example profile."""
        return cls(omega=11.0, alpha_1=10.0, alpha_2=0.002, **overrides)

    @property
    def threshold(self) -> float:
        """Return the instantaneous adaptive threshold in millivolts."""
        return self.omega + self.theta1 + self.theta2

    def _validate_state(self) -> None:
        """Reject invalid state and parameters before any mutation."""
        finite_values = (
            self.v,
            self.theta1,
            self.theta2,
            self.refractory_remaining,
            self.omega,
            self.tau_m,
            self.tau_1,
            self.tau_2,
            self.alpha_1,
            self.alpha_2,
            self.resistance,
            self.refractory_period,
            self.dt,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("MATNeuron state and parameters must be finite")
        if not (_VOLTAGE_MIN <= self.v <= _VOLTAGE_MAX):
            raise ValueError("MATNeuron voltage is outside the safety envelope")
        if not (-_THETA_MAX <= self.omega <= _THETA_MAX):
            raise ValueError("MATNeuron baseline threshold is outside the safety envelope")
        if not (0.0 <= self.theta1 <= _THETA_MAX and 0.0 <= self.theta2 <= _THETA_MAX):
            raise ValueError("MATNeuron threshold history is outside the safety envelope")
        if not (0.0 <= self.alpha_1 <= _THETA_MAX and 0.0 <= self.alpha_2 <= _THETA_MAX):
            raise ValueError("MATNeuron threshold increments are outside the safety envelope")
        if self.tau_m <= 0.0 or self.tau_1 <= 0.0 or self.tau_2 <= 0.0:
            raise ValueError("MATNeuron time constants must be positive")
        if self.resistance <= 0.0 or self.refractory_period < 0.0 or self.dt <= 0.0:
            raise ValueError(
                "MATNeuron resistance/timestep must be positive and refractory nonnegative"
            )
        if not (0.0 <= self.refractory_remaining <= self.refractory_period):
            raise ValueError("MATNeuron refractory state is outside its configured interval")

    def step(self, current: float) -> int:
        """Advance one paper-MAT* step and return ``1`` on a spike.

        Voltage is never reset. During the absolute refractory interval it
        continues to evolve, and a still-suprathreshold voltage may emit again
        as soon as that interval expires. Invalid candidates fail atomically.
        """
        if not math.isfinite(current):
            raise ValueError("MATNeuron input current must be finite")
        self._validate_state()

        v_candidate = self.v + self.dt * (-self.v + self.resistance * current) / self.tau_m
        theta1_candidate = self.theta1 * math.exp(-self.dt / self.tau_1)
        theta2_candidate = self.theta2 * math.exp(-self.dt / self.tau_2)
        refractory_candidate = max(0.0, self.refractory_remaining - self.dt)
        candidates = (
            v_candidate,
            theta1_candidate,
            theta2_candidate,
            refractory_candidate,
        )
        if not all(math.isfinite(value) for value in candidates):
            raise ValueError("MATNeuron candidate state must be finite")
        if not (_VOLTAGE_MIN <= v_candidate <= _VOLTAGE_MAX):
            raise ValueError("MATNeuron voltage candidate left the safety envelope")
        if not (0.0 <= theta1_candidate <= _THETA_MAX and 0.0 <= theta2_candidate <= _THETA_MAX):
            raise ValueError("MATNeuron threshold candidate left the safety envelope")

        spike = refractory_candidate == 0.0 and v_candidate >= (
            self.omega + theta1_candidate + theta2_candidate
        )
        if spike:
            theta1_after_spike = theta1_candidate + self.alpha_1
            theta2_after_spike = theta2_candidate + self.alpha_2
            if theta1_after_spike > _THETA_MAX or theta2_after_spike > _THETA_MAX:
                raise ValueError("MATNeuron post-spike threshold left the safety envelope")
            theta1_candidate = theta1_after_spike
            theta2_candidate = theta2_after_spike
            refractory_candidate = self.refractory_period

        self.v = v_candidate
        self.theta1 = theta1_candidate
        self.theta2 = theta2_candidate
        self.refractory_remaining = refractory_candidate
        return int(spike)

    def reset(self) -> None:
        """Restore zero-rest voltage, spike history, and refractory state."""
        self.v = 0.0
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.refractory_remaining = 0.0
