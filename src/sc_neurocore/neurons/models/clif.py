# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complementary LIF — dual positive/negative paths

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ComplementaryLIFNeuron:
    """Complementary LIF with separate positive and negative leaky paths.

    The public spike contract is ternary: +1 for positive-threshold crossing,
    -1 for negative-threshold crossing, and 0 for no spike.
    """

    v_pos: float = 0.0
    v_neg: float = 0.0
    tau: float = 10.0
    v_threshold: float = 1.0
    dt: float = 1.0
    alpha: float = field(init=False)

    _V_MAX: float = 1.0e12

    def __post_init__(self) -> None:
        self.alpha = self._validated_alpha()

    @staticmethod
    def _finite(value: float, name: str) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @classmethod
    def _positive(cls, value: float, name: str) -> float:
        value = cls._finite(value, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    def _validated_alpha(self) -> float:
        tau = self._positive(self.tau, "tau")
        dt = self._positive(self.dt, "dt")
        ratio = -dt / tau
        if ratio < -700.0:
            return 0.0
        alpha = math.exp(ratio)
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        return alpha

    def _validated_state(self) -> tuple[float, float, float]:
        v_pos = self._finite(self.v_pos, "v_pos")
        v_neg = self._finite(self.v_neg, "v_neg")
        if abs(v_pos) > self._V_MAX or abs(v_neg) > self._V_MAX:
            raise ValueError("CLIF membrane paths outside safety envelope")
        self._positive(self.v_threshold, "v_threshold")
        alpha = self._validated_alpha()
        return v_pos, v_neg, alpha

    def step(self, current: float) -> int:
        current = self._finite(current, "current")
        v_pos, v_neg, alpha = self._validated_state()
        inp_pos = max(current, 0.0)
        inp_neg = max(-current, 0.0)
        v_pos_next = alpha * v_pos + inp_pos
        v_neg_next = alpha * v_neg + inp_neg
        diff = v_pos_next - v_neg_next

        for value, name in (
            (v_pos_next, "v_pos candidate"),
            (v_neg_next, "v_neg candidate"),
            (diff, "difference candidate"),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if abs(v_pos_next) > self._V_MAX or abs(v_neg_next) > self._V_MAX:
            raise ValueError("CLIF membrane candidate outside safety envelope")

        self.alpha = alpha
        if diff >= self.v_threshold:
            self.v_pos = 0.0
            self.v_neg = 0.0
            return 1
        if diff <= -self.v_threshold:
            self.v_pos = 0.0
            self.v_neg = 0.0
            return -1
        self.v_pos = v_pos_next
        self.v_neg = v_neg_next
        return 0

    def reset(self) -> None:
        self.v_pos, self.v_neg = 0.0, 0.0
