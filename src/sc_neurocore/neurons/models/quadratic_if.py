# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quadratic Integrate-and-Fire — canonical Type-I excitability

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class QuadraticIFNeuron:
    """Quadratic Integrate-and-Fire — canonical Type-I excitability.

    dv/dt = v² + I
    Reset when v >= v_peak.

    Reference: Latham, P.E. et al. (2000). J. Neurophysiol. 83:808–827.
    """

    v: float = -1.0
    v_reset: float = -1.0
    v_peak: float = 1.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        for field in ("v", "v_reset", "v_peak"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if self.v >= self.v_peak:
            raise ValueError("v must be below v_peak")
        if self.v_reset >= self.v_peak:
            raise ValueError("v_peak must be greater than v_reset")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")

    def _exact_candidate(self, current: float) -> tuple[float, bool]:
        if current > 0.0:
            root_i = math.sqrt(current)
            phase = math.atan(self.v / root_i)
            peak_phase = math.atan(self.v_peak / root_i)
            next_phase = phase + root_i * self.dt
            if next_phase >= peak_phase or next_phase >= math.pi / 2.0:
                return self.v_reset, True
            return root_i * math.tan(next_phase), False
        if current == 0.0:
            denominator = 1.0 - self.v * self.dt
            if denominator <= 0.0:
                return self.v_reset, True
            next_v = self.v / denominator
            return (self.v_reset, True) if next_v >= self.v_peak else (next_v, False)

        root_i = math.sqrt(-current)
        if math.isclose(self.v, -root_i, rel_tol=0.0, abs_tol=1e-15):
            return self.v, False
        numerator_ratio = (self.v - root_i) / (self.v + root_i)
        try:
            evolved_ratio = numerator_ratio * math.exp(2.0 * root_i * self.dt)
        except OverflowError:
            return math.nan, False
        denominator = 1.0 - evolved_ratio
        if numerator_ratio < 1.0 <= evolved_ratio or math.isclose(
            denominator, 0.0, rel_tol=0.0, abs_tol=1e-15
        ):
            return self.v_reset, True
        next_v = root_i * (1.0 + evolved_ratio) / denominator
        return (self.v_reset, True) if next_v >= self.v_peak else (next_v, False)

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        next_v, spiked = self._exact_candidate(current)
        if not math.isfinite(next_v):
            raise ValueError("exact-flow candidate must be finite")
        self.v = next_v
        return int(spiked)

    def reset(self) -> None:
        self.v = self.v_reset
