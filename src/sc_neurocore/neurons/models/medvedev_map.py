# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Medvedev 2005 — 1D piecewise-monotone spiking map

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MedvedevMapNeuron:
    """Medvedev 2005 — 1D piecewise-monotone spiking map."""

    x: float = 0.0
    alpha: float = 3.5
    beta: float = 0.5
    x_threshold: float = 0.9

    def step(self, current: float = 0.0) -> int:
        x_prev = self.x
        if self.x < self.beta:
            self.x = self.alpha * self.x + current
        else:
            self.x = self.alpha * (1.0 - self.x) + current
        self.x = self.x % 1.0
        return 1 if (self.x >= self.x_threshold and x_prev < self.x_threshold) else 0

    def reset(self) -> None:
        self.x = 0.0


# ── HARDWARE-SPECIFIC ──────────────────────────────────────────────
