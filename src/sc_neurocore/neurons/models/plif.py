# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fang et al. 2021 — Parametric LIF (PLIF) with learnable

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ParametricLIFNeuron:
    """Fang et al. 2021 — Parametric LIF (PLIF) with learnable decay.

    V(t+1) = alpha * V(t) * (1 - spike(t)) + I(t)
    alpha  = sigmoid(a)    (learnable parameter)
    spike  = Theta(V - threshold)

    Reference: Fang, W. et al. (2021). Proc. AAAI Conf. Artif. Intell. 35(3):2661–2669.
    """

    v: float = 0.0
    a: float = 0.0
    threshold: float = 1.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        for name in ("v", "a"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not math.isfinite(self.threshold) or self.threshold <= 0.0:
            raise ValueError("threshold must be finite and positive")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")

    @property
    def alpha(self) -> float:
        if self.a >= 0.0:
            z = math.exp(-self.a)
            return 1.0 / (1.0 + z)
        z = math.exp(self.a)
        return z / (1.0 + z)

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()
        spike = 1 if self.v >= self.threshold else 0
        next_v = self.alpha * self.v * (1 - spike) + current
        if not math.isfinite(next_v):
            raise ValueError("voltage candidate must remain finite")
        self.v = next_v
        return 1 if next_v >= self.threshold else 0

    def reset(self) -> None:
        self.v = 0.0

    def _validate_runtime_state(self) -> None:
        for name in ("v", "a"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"runtime {name} must be finite")
        if not math.isfinite(self.threshold) or self.threshold <= 0.0:
            raise ValueError("runtime threshold must be finite and positive")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("runtime dt must be finite and positive")
