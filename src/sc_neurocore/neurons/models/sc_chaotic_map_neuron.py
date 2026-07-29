# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — preserved SC engineering two-state chaotic map

"""SC-NeuroCore's historical two-state sigmoid-gated map.

This is a project-designed engineering model. It was formerly exposed under
the Aihara paper identity, but its recurrence is not Aihara's Eqs. 10–12. The
distinct class preserves the useful SC model without conflating provenance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

SCChaoticMapResult = dict[str, npt.NDArray[np.float64] | float | int]


@dataclass
class SCChaoticMapNeuron:
    """Two-state sigmoid-gated SC chaotic map with bounded state."""

    x: float = 0.0
    y: float = 0.0
    k_f: float = 0.7
    k_s: float = 0.95
    alpha: float = 2.0
    delta: float = 0.05
    x_threshold: float = 0.5

    def __post_init__(self) -> None:
        for name in ("x", "y", "k_f", "k_s", "alpha", "delta", "x_threshold"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        if self.k_f < 0.0:
            raise ValueError("k_f must be non-negative")
        if self.delta < 0.0:
            raise ValueError("delta must be non-negative")

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0.0:
            return 1.0 / (1.0 + math.exp(-value))
        exponential = math.exp(value)
        return exponential / (1.0 + exponential)

    def step(self, current: float = 0.0) -> int:
        """Commit one simultaneous bounded update and return an upward crossing."""
        try:
            drive = float(current)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("current must be numeric") from exc
        if not math.isfinite(drive):
            raise ValueError("current must be finite")
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise FloatingPointError("SC chaotic map state must be finite")
        x_previous = self.x
        x_candidate = self.k_f * self.x * self._sigmoid(self.x + self.alpha) - self.y + drive
        y_candidate = self.k_s * self.y + self.delta * self.x
        if not math.isfinite(x_candidate) or not math.isfinite(y_candidate):
            raise FloatingPointError("SC chaotic map candidate state became non-finite")
        self.x = max(-10.0, min(10.0, x_candidate))
        self.y = max(-10.0, min(10.0, y_candidate))
        return int(x_previous < self.x_threshold <= self.x)

    def simulate(
        self, current: npt.ArrayLike, *, backend: str = "auto"
    ) -> SCChaoticMapResult:
        """Run an atomic batch on a parity-checked maintained backend."""
        from sc_neurocore.accel.sc_chaotic_map import simulate_sc_chaotic_map

        result = simulate_sc_chaotic_map(
            self.x,
            self.y,
            self.k_f,
            self.k_s,
            self.alpha,
            self.delta,
            self.x_threshold,
            current,
            backend=backend,
        )
        self.x = float(cast(float, result["x_final"]))
        self.y = float(cast(float, result["y_final"]))
        return result

    def reset(self) -> None:
        """Clear both states while preserving the configured parameters."""
        self.x = 0.0
        self.y = 0.0


__all__ = ["SCChaoticMapNeuron", "SCChaoticMapResult"]
