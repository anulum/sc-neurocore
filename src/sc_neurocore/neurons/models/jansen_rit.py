# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Jansen & Rit 1995 — neural mass model for EEG generation

from __future__ import annotations

from dataclasses import dataclass
import math


_STATE_NAMES = ("y0", "y3", "y1", "y4", "y2", "y5")
_PARAM_NAMES = ("a_exc", "b_exc", "a_rate", "b_rate", "c", "e0", "v0", "r", "dt")
_STRICTLY_POSITIVE_PARAMS = ("a_exc", "b_exc", "a_rate", "b_rate", "e0", "r", "dt")


@dataclass
class JansenRitUnit:
    """Jansen & Rit 1995 — neural mass model for EEG generation.

    6 ODEs: 3 populations (pyramidal, excitatory, inhibitory) x 2 states.

    Reference: Jansen, B.H. & Rit, V.G. (1995). Biol. Cybern. 73:357–366.
    """

    y0: float = 0.0
    y3: float = 0.0
    y1: float = 0.0
    y4: float = 0.0
    y2: float = 0.0
    y5: float = 0.0
    a_exc: float = 3.25
    b_exc: float = 22.0
    a_rate: float = 100.0
    b_rate: float = 50.0
    c: float = 135.0
    e0: float = 2.5
    v0: float = 6.0
    r: float = 0.56
    dt: float = 0.001

    def __post_init__(self) -> None:
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.c < 0.0:
            raise ValueError("c must be non-negative")

    @staticmethod
    def _require_finite(name: str, value: float) -> float:
        out = float(value)
        if not math.isfinite(out):
            raise ValueError(f"{name} must be finite")
        return out

    def _validate_state(self, values: tuple[float, ...] | None = None) -> tuple[float, ...]:
        state = values if values is not None else tuple(getattr(self, name) for name in _STATE_NAMES)
        checked = tuple(self._require_finite(name, value) for name, value in zip(_STATE_NAMES, state))
        if len(checked) != len(_STATE_NAMES):
            raise ValueError("Jansen-Rit state vector has invalid dimension")
        return checked

    def _sigmoid(self, x: float) -> float:
        drive = self._require_finite("sigmoid input", x)
        exponent = self.r * (self.v0 - drive)
        if exponent >= 0.0:
            exp_neg = math.exp(-exponent)
            return 2.0 * self.e0 * exp_neg / (1.0 + exp_neg)
        return 2.0 * self.e0 / (1.0 + math.exp(exponent))

    def step(self, p_ext: float = 220.0) -> float:
        p_ext = self._require_finite("p_ext", p_ext)
        y0, y3, y1, y4, y2, y5 = self._validate_state()

        s1 = self._sigmoid(y1 - y2)
        s0 = self._sigmoid(self.c * 0.8 * y0)
        s2 = self._sigmoid(self.c * 0.25 * y0)
        dy0 = y3
        dy3 = self.a_exc * self.a_rate * s1 - 2.0 * self.a_rate * y3 - self.a_rate**2 * y0
        dy1 = y4
        dy4 = (
            self.a_exc * self.a_rate * (p_ext + self.c * 0.8 * s0)
            - 2.0 * self.a_rate * y4
            - self.a_rate**2 * y1
        )
        dy2 = y5
        dy5 = (
            self.b_exc * self.b_rate * self.c * 0.25 * s2
            - 2.0 * self.b_rate * y5
            - self.b_rate**2 * y2
        )
        candidate = (
            y0 + dy0 * self.dt,
            y3 + dy3 * self.dt,
            y1 + dy1 * self.dt,
            y4 + dy4 * self.dt,
            y2 + dy2 * self.dt,
            y5 + dy5 * self.dt,
        )
        self.y0, self.y3, self.y1, self.y4, self.y2, self.y5 = self._validate_state(
            candidate
        )
        return self.y1 - self.y2

    def reset(self) -> None:
        self.y0 = self.y1 = self.y2 = self.y3 = self.y4 = self.y5 = 0.0
