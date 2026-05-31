# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh 1976 / Rinzel 1987 — FHN + slow variable for

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FitzHughRinzelNeuron:
    """FitzHugh 1976 / Rinzel 1987 — FHN + slow variable for bursting.

    Reference: Rinzel, J. (1987). In: Mathematical Topics in Population Biology. Springer, pp. 267–281.
    """

    v: float = -1.0
    w: float = -0.5
    y: float = 0.0
    a: float = 0.7
    b: float = 0.8
    c: float = -0.775
    d: float = 1.0
    delta: float = 0.08
    mu: float = 0.0001
    dt: float = 0.1
    v_threshold: float = 1.0

    def __post_init__(self) -> None:
        self._validate_numeric_contract()

    def _numeric_fields(self) -> tuple[tuple[str, float], ...]:
        return (
            ("v", self.v),
            ("w", self.w),
            ("y", self.y),
            ("a", self.a),
            ("b", self.b),
            ("c", self.c),
            ("d", self.d),
            ("delta", self.delta),
            ("mu", self.mu),
            ("dt", self.dt),
            ("v_threshold", self.v_threshold),
        )

    def _validate_numeric_contract(self) -> None:
        for name, value in self._numeric_fields():
            if not math.isfinite(value):
                raise ValueError(f"FitzHugh-Rinzel parameter {name} must be finite")
        for name, value in (("delta", self.delta), ("mu", self.mu), ("dt", self.dt)):
            if value <= 0.0:
                raise ValueError(f"FitzHugh-Rinzel parameter {name} must be positive")

    def _derivatives(
        self, v: float, w: float, y: float, current: float
    ) -> tuple[float, float, float]:
        if not all(math.isfinite(value) for value in (v, w, y, current)):
            raise FloatingPointError("FitzHugh-Rinzel runtime state and current must be finite")
        try:
            dv = v - v**3 / 3.0 - w + y + current
            dw = self.delta * (self.a + v - self.b * w)
            dy = self.mu * (self.c - v - self.d * y)
        except OverflowError as exc:
            raise FloatingPointError("FitzHugh-Rinzel derivative overflow") from exc
        if not all(math.isfinite(value) for value in (dv, dw, dy)):
            raise FloatingPointError("FitzHugh-Rinzel derivative must be finite")
        return dv, dw, dy

    def step(self, current: float) -> int:
        """Advance the model by one simultaneous-Euler step."""

        self._validate_numeric_contract()
        v_prev = self.v
        dv, dw, dy = self._derivatives(self.v, self.w, self.y, float(current))
        next_v = self.v + dv * self.dt
        next_w = self.w + dw * self.dt
        next_y = self.y + dy * self.dt
        if not all(math.isfinite(value) for value in (next_v, next_w, next_y)):
            raise FloatingPointError("FitzHugh-Rinzel candidate state must be finite")
        self.v = next_v
        self.w = next_w
        self.y = next_y
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v, self.w, self.y = -1.0, -0.5, 0.0
