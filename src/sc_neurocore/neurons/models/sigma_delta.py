# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Yoon 2017 — event-driven sigma-delta encoding

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SigmaDeltaNeuron:
    """Yoon 2017 — event-driven sigma-delta encoding."""

    sigma: float = 0.0
    v_threshold: float = 1.0

    def step(self, current: float) -> int:
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
