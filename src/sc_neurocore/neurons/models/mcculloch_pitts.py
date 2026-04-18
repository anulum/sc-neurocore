# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — McCulloch & Pitts 1943 — binary threshold neuron

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class McCullochPittsNeuron:
    """McCulloch & Pitts 1943 — binary threshold neuron.

    y = 1 if sum(w_i * x_i) >= theta, else 0.

    Reference: McCulloch, W.S. & Pitts, W. (1943). Bull. Math. Biophys. 5:115–133.
    """

    theta: float = 1.0

    def step(self, weighted_input: float) -> int:
        return 1 if weighted_input >= self.theta else 0

    def reset(self) -> None:
        pass
