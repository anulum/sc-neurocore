# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Oster, Douglas & Liu 2009 — winner-take-all with lateral

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


@dataclass
class LeakyCompeteFireNeuron:
    """Oster, Douglas & Liu 2009 — winner-take-all with lateral inhibition.

    Reference: Oster, M. et al. (2009). Neural Comput. 21(9):2437–2465.
    """

    n_units: int = 4
    v: list[float] = field(default_factory=lambda: [0.0] * 4)
    tau: float = 10.0
    v_threshold: float = 1.0
    w_inh: float = 0.5
    dt: float = 1.0

    def __post_init__(self) -> None:
        if self.n_units <= 0:
            raise ValueError("n_units must be positive")
        if not isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("tau must be finite and positive")
        if not isfinite(self.v_threshold):
            raise ValueError("v_threshold must be finite")
        if not isfinite(self.w_inh) or self.w_inh < 0.0:
            raise ValueError("w_inh must be finite and non-negative")
        if not isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.v = [0.0] * self.n_units

    def step(self, currents: list[float] | float) -> list[int]:
        if isinstance(currents, (int, float)):
            currents = [currents] * self.n_units
        if len(currents) != self.n_units:
            raise ValueError(f"currents must have length {self.n_units}")
        if any(not isfinite(current) for current in currents):
            raise ValueError("currents must contain only finite values")
        spikes = [0] * self.n_units
        for i in range(self.n_units):
            self.v[i] += (-self.v[i] + currents[i]) / self.tau * self.dt
        for i in range(self.n_units):
            if self.v[i] >= self.v_threshold:
                spikes[i] = 1
                self.v[i] = 0.0
                for j in range(self.n_units):
                    if j != i:
                        self.v[j] -= self.w_inh
                        self.v[j] = max(0.0, self.v[j])
        return spikes

    def reset(self) -> None:
        self.v = [0.0] * self.n_units
