# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Poisson spike generator — stochastic firing at rate λ

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PoissonNeuron:
    """Poisson spike generator — stochastic firing at rate λ.

    P(spike in dt) = 1 - exp(-λ · dt). Essential for input layer generation.

    Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §7.2.
    """

    rate_hz: float = 100.0
    dt_ms: float = 1.0
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._validate_runtime_state()
        self._probability(self.rate_hz)
        self._rng = np.random.default_rng()

    def _validate_runtime_state(self) -> None:
        if not math.isfinite(self.rate_hz) or self.rate_hz < 0.0:
            raise ValueError("rate_hz must be finite and non-negative")
        if not math.isfinite(self.dt_ms) or self.dt_ms <= 0.0:
            raise ValueError("dt_ms must be finite and positive")

    def _probability(self, rate_hz: float) -> float:
        if not math.isfinite(rate_hz) or rate_hz < 0.0:
            raise ValueError("rate_hz must be finite and non-negative")
        hazard = rate_hz * self.dt_ms / 1000.0
        if not math.isfinite(hazard) or hazard < 0.0:
            raise ValueError("interval hazard must be finite and non-negative")
        probability = -math.expm1(-hazard)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("spike probability must remain finite and bounded")
        return probability

    def step(self, rate_override: float = -1.0) -> int:
        if not math.isfinite(rate_override):
            raise ValueError("rate_override must be finite")
        self._validate_runtime_state()
        r = self.rate_hz if rate_override < 0 else rate_override
        p = self._probability(r)
        return 1 if self._rng.random() < p else 0

    def reset(self) -> None:
        pass
