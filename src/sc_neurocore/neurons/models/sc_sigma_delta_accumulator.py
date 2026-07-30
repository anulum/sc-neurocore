# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained project bipolar sigma-delta accumulator

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SCSigmaDeltaAccumulatorNeuron:
    """Retained bipolar accumulate/one-quantum-subtract recurrence.

    This is exactly the historical project behavior formerly exposed as
    ``SigmaDeltaNeuron``. It emits at most one signed event per sample and
    carries any remaining threshold excess forward. It is project-defined and
    intentionally carries no external paper attribution.
    """

    sigma: float = 0.0
    v_threshold: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma):
            raise ValueError("sigma must be finite")
        if not math.isfinite(self.v_threshold) or self.v_threshold <= 0.0:
            raise ValueError("v_threshold must be finite and positive")

    def step(self, current: float) -> int:
        """Advance the frozen bipolar project recurrence by one sample."""
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        candidate = self.sigma + current
        if not math.isfinite(candidate):
            raise ValueError("sigma update must remain finite")
        event = 0
        if candidate >= self.v_threshold:
            candidate -= self.v_threshold
            event = 1
        elif candidate <= -self.v_threshold:
            candidate += self.v_threshold
            event = -1
        if not math.isfinite(candidate):
            raise ValueError("sigma update must remain finite")
        self.sigma = candidate
        return event

    def reset(self) -> None:
        """Clear the accumulator while retaining its threshold."""
        self.sigma = 0.0
