# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LeakyCompeteFireNeuron:
    """Oster, Douglas & Liu 2009 — winner-take-all with lateral inhibition."""

    n_units: int = 4
    v: list = field(default_factory=lambda: [0.0] * 4)
    tau: float = 10.0
    v_threshold: float = 1.0
    w_inh: float = 0.5
    dt: float = 1.0

    def __post_init__(self):
        self.v = [0.0] * self.n_units

    def step(self, currents: list) -> list:
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

    def reset(self):
        self.v = [0.0] * self.n_units
