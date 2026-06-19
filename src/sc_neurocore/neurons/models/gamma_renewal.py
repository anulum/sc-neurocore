# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Gamma renewal process neuron. Keat et al. 2001

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GammaRenewalNeuron:
    """Gamma renewal process neuron. Keat et al. 2001.

    ISI ~ Gamma(k, k/rate). Hazard h(t) evaluated at elapsed time
    since last spike. P(spike in dt) = 1 - exp(-h(t)*dt).

    Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §7.4.
    """

    rate_hz: float = 50.0
    shape_k: int = 3
    dt_ms: float = 1.0
    _time_since_spike: float = 0.0
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.rate_hz) or self.rate_hz < 0.0:
            raise ValueError("rate_hz must be finite and non-negative")
        if not isinstance(self.shape_k, int) or self.shape_k <= 0:
            raise ValueError("shape_k must be a positive integer")
        if not math.isfinite(self.dt_ms) or self.dt_ms <= 0.0:
            raise ValueError("dt_ms must be finite and positive")
        if not math.isfinite(self._time_since_spike) or self._time_since_spike < 0.0:
            raise ValueError("time_since_spike must be finite and non-negative")
        self._rng = np.random.default_rng()

    def step(self, rate_override: float = -1.0) -> int:
        if not math.isfinite(rate_override):
            raise ValueError("rate_override must be finite")
        r = self.rate_hz if rate_override < 0 else rate_override
        self._time_since_spike += self.dt_ms / 1000.0
        if r == 0.0:
            return 0
        t = self._time_since_spike
        k = self.shape_k
        lam = k * r
        # Gamma hazard: h(t) = f(t) / (1 - F(t)) approximated via scipy-free form
        # f(t) = lam^k * t^(k-1) * exp(-lam*t) / Gamma(k)
        if t < 1e-12:
            return 0
        log_f = k * np.log(lam) + (k - 1) * np.log(t) - lam * t - _log_gamma_int(k)
        f = np.exp(np.clip(log_f, -50.0, 50.0))
        # Survival approximated as 1 - regularized_gamma (use upper incomplete)
        survival = _gamma_survival(k, lam * t)
        if survival < 1e-15:
            survival = 1e-15
        hazard = f / survival
        p = -math.expm1(-(hazard * self.dt_ms / 1000.0))
        if self._rng.random() < p:
            self._time_since_spike = 0.0
            return 1
        return 0

    def reset(self) -> None:
        self._time_since_spike = 0.0


def _log_gamma_int(k: int) -> float:
    """ln(Gamma(k)) for positive integer k = ln((k-1)!)."""
    return sum(np.log(i) for i in range(1, k)) if k > 1 else 0.0


def _gamma_survival(k: int, x: float) -> float:
    """P(X > x) for Gamma(k, 1) via series for upper incomplete gamma."""
    if x < 0:
        return 1.0
    s = 1.0
    term = 1.0
    for i in range(1, k):
        term *= x / i
        s += term
    return float(np.exp(-x) * s)
