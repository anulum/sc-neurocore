# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Benda & Herz 2003 — phenomenological spike-frequency

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BendaHerzNeuron:
    """Benda & Herz 2003 — phenomenological spike-frequency adaptation.

    f = f_onset(I - A)          instantaneous f-I curve
    dA/dt = -A/tau_a + delta_a * f
    f_onset(x) = f_max / (1 + exp(-beta*(x - I_half)))
    """

    a: float = 0.0
    f_max: float = 200.0
    beta: float = 0.1
    i_half: float = 5.0
    tau_a: float = 100.0
    delta_a: float = 0.5
    dt: float = 1.0
    _rng: object = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()

    def _f_onset(self, x: float) -> float:
        return self.f_max / (1.0 + np.exp(-self.beta * (x - self.i_half)))

    def step(self, current: float) -> int:
        rate = self._f_onset(current - self.a)
        self.a += (-self.a / self.tau_a + self.delta_a * rate) * self.dt
        p = rate * self.dt / 1000.0
        return 1 if self._rng.random() < min(p, 1.0) else 0

    def reset(self) -> None:
        self.a = 0.0
