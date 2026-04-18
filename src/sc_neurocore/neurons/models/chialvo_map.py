# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chialvo 1995 — 2D discrete map neuron

from __future__ import annotations

from dataclasses import dataclass

from sc_neurocore.utils.numerics import safe_exp


@dataclass
class ChialvoMapNeuron:
    """Chialvo 1995 — 2D discrete map neuron.

    x[n+1] = x²·exp(y-x) + k + I
    y[n+1] = a·y - b·x + c
    """

    x: float = 0.0
    y: float = 0.0
    a: float = 0.89
    b: float = 0.6
    c: float = 0.28
    k: float = 0.04
    x_threshold: float = 1.0

    def step(self, current: float = 0.0) -> int:
        x_prev = self.x
        x_new = self.x**2 * safe_exp(self.y - self.x) + self.k + current
        y_new = self.a * self.y - self.b * self.x + self.c
        self.x = x_new
        self.y = y_new
        return 1 if (self.x >= self.x_threshold and x_prev < self.x_threshold) else 0

    def reset(self) -> None:
        self.x, self.y = 0.0, 0.0
