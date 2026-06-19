# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Yao et al. 2022 NeurIPS — LIF with learnable gates

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GatedLIFNeuron:
    """Yao et al. 2022 NeurIPS — LIF with learnable gates.

    Reference: Yao, M. et al. (2022). Proc. NeurIPS 35:19606–19618.
    """

    v: float = 0.0
    gate_v: float = 0.9
    gate_i: float = 1.0
    v_threshold: float = 1.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.v):
            raise ValueError("v must be finite")
        if not math.isfinite(self.gate_i):
            raise ValueError("gate_i must be finite")
        if not math.isfinite(self.gate_v) or not 0.0 <= self.gate_v <= 1.0:
            raise ValueError("gate_v must be finite and within [0, 1]")
        for field in ("v_threshold", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self.v = self.gate_v * self.v + self.gate_i * current
        if self.v >= self.v_threshold:
            self.v -= self.v_threshold
            return 1
        return 0

    def reset(self) -> None:
        self.v = 0.0


# ── POPULATION / RATE / NEURAL MASS ───────────────────────────────
