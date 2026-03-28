# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Poisson spike generator — stochastic firing at rate λ

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PoissonNeuron:
    """Poisson spike generator — stochastic firing at rate λ.

    P(spike in dt) = λ · dt. Essential for input layer generation.
    """

    rate_hz: float = 100.0
    dt_ms: float = 1.0
    _rng: object = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()

    def step(self, rate_override: float = -1.0) -> int:
        r = self.rate_hz if rate_override < 0 else rate_override
        p = r * self.dt_ms / 1000.0
        return 1 if self._rng.random() < p else 0

    def reset(self) -> None:
        pass
