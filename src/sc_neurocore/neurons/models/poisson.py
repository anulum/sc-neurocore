# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Poisson spike generator — stochastic firing at rate λ

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class PoissonNeuron:
    """Poisson spike generator — stochastic firing at rate λ.

    P(spike in dt) = λ · dt. Essential for input layer generation.

    Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §7.2.
    """

    rate_hz: float = 100.0
    dt_ms: float = 1.0
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.rate_hz) or self.rate_hz < 0.0:
            raise ValueError("rate_hz must be finite and non-negative")
        if not math.isfinite(self.dt_ms) or self.dt_ms <= 0.0:
            raise ValueError("dt_ms must be finite and positive")
        self._probability(self.rate_hz)
        self._rng = np.random.default_rng()

    def _probability(self, rate_hz: float) -> float:
        p = rate_hz * self.dt_ms / 1000.0
        if p > 1.0:
            raise ValueError("spike probability must not exceed one")
        return p

    def step(self, rate_override: float = -1.0) -> int:
        if not math.isfinite(rate_override):
            raise ValueError("rate_override must be finite")
        r = self.rate_hz if rate_override < 0 else rate_override
        p = self._probability(r)
        return 1 if self._rng.random() < p else 0

    def reset(self) -> None:
        pass
