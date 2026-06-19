# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chialvo 1995 — 2D discrete map neuron

from __future__ import annotations

import math
from dataclasses import dataclass

from sc_neurocore.utils.numerics import safe_exp


@dataclass
class ChialvoMapNeuron:
    """Chialvo 1995 — 2D discrete map neuron.

    x[n+1] = x²·exp(y-x) + k + I
    y[n+1] = a·y - b·x + c

    Reference: Chialvo, D.R. (1995). Chaos, Solitons & Fractals 5:461–479.
    """

    x: float = 0.0
    y: float = 0.0
    a: float = 0.89
    b: float = 0.6
    c: float = 0.28
    k: float = 0.04
    x_threshold: float = 1.0

    def __post_init__(self) -> None:
        for name in ("x", "y", "a", "b", "c", "k", "x_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)

    @staticmethod
    def _validate_state(x: float, y: float) -> tuple[float, float]:
        x_value = float(x)
        y_value = float(y)
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            raise FloatingPointError("Chialvo map state must be finite")
        return x_value, y_value

    def step(self, current: float = 0.0) -> int:
        drive = float(current)
        if not math.isfinite(drive):
            raise ValueError("current must be finite")

        x, y = self._validate_state(self.x, self.y)
        x_prev = x
        try:
            x_new = x**2 * safe_exp(y - x) + self.k + drive
        except OverflowError as exc:
            raise FloatingPointError("Chialvo map quadratic term overflowed") from exc
        y_new = self.a * y - self.b * x + self.c
        if not math.isfinite(x_new) or not math.isfinite(y_new):
            raise FloatingPointError("Chialvo map candidate state became non-finite")
        self.x = x_new
        self.y = y_new
        return 1 if (self.x >= self.x_threshold and x_prev < self.x_threshold) else 0

    def reset(self) -> None:
        self.x, self.y = 0.0, 0.0
