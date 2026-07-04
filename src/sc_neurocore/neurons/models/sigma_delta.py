# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Yoon 2017 — event-driven sigma-delta encoding

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SigmaDeltaNeuron:
    """Yoon 2017 — event-driven sigma-delta encoding.

    Reference: Yoon, Y. C. (2017). LIF and simplified SRM neurons encode
    signals into spikes via a form of asynchronous pulse sigma-delta
    modulation. IEEE Trans. Neural Netw. Learn. Syst. (DOI 10.1109/tnnls.2016.2526029).
    """

    sigma: float = 0.0
    v_threshold: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma):
            raise ValueError("sigma must be finite")
        if not math.isfinite(self.v_threshold) or self.v_threshold <= 0:
            raise ValueError("v_threshold must be finite and positive")

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")

        self.sigma += current
        if self.sigma >= self.v_threshold:
            self.sigma -= self.v_threshold
            return 1
        elif self.sigma <= -self.v_threshold:
            self.sigma += self.v_threshold
            return -1
        return 0

    def reset(self) -> None:
        self.sigma = 0.0
