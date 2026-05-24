# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Benda & Herz 2003 — phenomenological spike-frequency

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class BendaHerzNeuron:
    """Benda & Herz 2003 — phenomenological spike-frequency adaptation.

    f = f_onset(I - A)          instantaneous f-I curve
    dA/dt = -A/tau_a + delta_a * f
    f_onset(x) = f_max / (1 + exp(-beta*(x - I_half)))

    Reference: Benda, J. & Herz, A.V.M. (2003). Neural Comput. 15:2523–2564.
    """

    a: float = 0.0
    f_max: float = 200.0
    beta: float = 0.1
    i_half: float = 5.0
    tau_a: float = 100.0
    delta_a: float = 0.5
    dt: float = 1.0
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.a) or self.a < 0.0:
            raise ValueError("a must be finite and non-negative")
        for name in ("f_max", "beta", "tau_a", "dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.i_half):
            raise ValueError("i_half must be finite")
        if not math.isfinite(self.delta_a) or self.delta_a < 0.0:
            raise ValueError("delta_a must be finite and non-negative")
        self._rng = np.random.default_rng()

    def _f_onset(self, x: float) -> float:
        z = self.beta * (x - self.i_half)
        if z >= 0.0:
            return self.f_max / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return self.f_max * exp_z / (1.0 + exp_z)

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")

        rate = self._f_onset(current - self.a)
        p = rate * self.dt / 1000.0
        if not math.isfinite(rate) or not math.isfinite(p):
            raise ValueError("spike probability must be finite")
        if p > 1.0:
            raise ValueError("spike probability must not exceed one")
        next_a = self.a + (-self.a / self.tau_a + self.delta_a * rate) * self.dt
        if not math.isfinite(next_a) or next_a < 0.0:
            raise ValueError("adaptation update must be finite and non-negative")
        self.a = next_a
        return 1 if self._rng.random() < p else 0

    def reset(self) -> None:
        self.a = 0.0
