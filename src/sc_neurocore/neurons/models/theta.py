# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Theta neuron — canonical Type-I on the unit circle

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class ThetaNeuron:
    """Theta neuron — canonical Type-I on the unit circle.

    dθ/dt = (1 - cos θ) + (1 + cos θ) · I
    Spike when θ crosses π.
    Ermentrout & Kopell 1986.

    Reference: Ermentrout, G.B. & Kopell, N. (1986). SIAM J. Appl. Math. 46:233–253.
    """

    theta: float = 0.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        if not math.isfinite(self.theta):
            raise ValueError("theta must be finite")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.theta = self._wrap_phase(self.theta)

    @staticmethod
    def _wrap_phase(theta: float) -> float:
        return ((theta + math.pi) % (2.0 * math.pi)) - math.pi

    def _exact_candidate(self, current: float) -> tuple[float, bool]:
        y = math.tan(self.theta / 2.0)
        if current > 0.0:
            root_i = math.sqrt(current)
            phase = math.atan(y / root_i)
            next_phase = phase + root_i * self.dt
            spiked = next_phase >= math.pi / 2.0
            if math.isclose(math.cos(next_phase), 0.0, rel_tol=0.0, abs_tol=1e-15):
                return -math.pi, spiked
            return self._wrap_phase(2.0 * math.atan(root_i * math.tan(next_phase))), spiked
        if current == 0.0:
            denominator = 1.0 - y * self.dt
            if math.isclose(denominator, 0.0, rel_tol=0.0, abs_tol=1e-15):
                return -math.pi, True
            next_y = y / denominator
            return self._wrap_phase(2.0 * math.atan(next_y)), denominator <= 0.0

        root_i = math.sqrt(-current)
        if math.isclose(y, -root_i, rel_tol=0.0, abs_tol=1e-15):
            return self.theta, False
        numerator_ratio = (y - root_i) / (y + root_i)
        try:
            evolved_ratio = numerator_ratio * math.exp(2.0 * root_i * self.dt)
        except OverflowError:
            return math.nan, False
        denominator = 1.0 - evolved_ratio
        spiked = numerator_ratio < 1.0 <= evolved_ratio or math.isclose(
            denominator,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        if spiked and math.isclose(denominator, 0.0, rel_tol=0.0, abs_tol=1e-15):
            return -math.pi, True
        next_y = root_i * (1.0 + evolved_ratio) / denominator
        return self._wrap_phase(2.0 * math.atan(next_y)), spiked

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()

        next_theta, spiked = self._exact_candidate(current)
        if not math.isfinite(next_theta):
            raise ValueError("exact-flow candidate must be finite")
        self.theta = self._wrap_phase(next_theta)
        return int(spiked)

    def reset(self) -> None:
        self.theta = 0.0

    def _validate_runtime_state(self) -> None:
        if not math.isfinite(self.theta):
            raise ValueError("runtime phase state must be finite")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("runtime dt must be finite and positive")
