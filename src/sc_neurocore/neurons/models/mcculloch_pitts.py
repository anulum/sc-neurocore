# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class McCullochPittsNeuron:
    """McCulloch & Pitts 1943 — binary threshold neuron.

    y = 1 if sum(w_i * x_i) >= theta, else 0.
    """

    theta: float = 1.0

    def step(self, weighted_input: float) -> int:
        return 1 if weighted_input >= self.theta else 0

    def reset(self):
        pass
