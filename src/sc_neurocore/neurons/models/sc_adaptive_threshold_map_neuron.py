# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — project adaptive-threshold sigmoid map

"""SC-NeuroCore's retained two-state adaptive-threshold map.

This repository-designed recurrence was formerly exposed under the unsupported
``KilincBhattMapNeuron`` identity. It is useful, but it is not the one-state
Nagumo–Sato model and is not an equation from the Aihara sources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

SCAdaptiveThresholdMapResult = dict[str, npt.NDArray[np.float64] | float | int]


@dataclass
class SCAdaptiveThresholdMapNeuron:
    """Bounded SC sigmoid map with a slow adaptive threshold.

    The simultaneous update is

    ``x' = clamp(-x + k*sigmoid(4*(x-theta)) + current, -5, 5)``
    ``theta' = clamp(beta*theta + gamma*H(x-theta_spike), -5, 5)``.

    The returned event is an upward crossing of ``x_threshold`` by ``x'``.
    """

    x: float = 0.0
    theta: float = 0.0
    k: float = 1.5
    beta: float = 0.95
    gamma: float = 0.3
    theta_spike: float = 0.8
    x_threshold: float = 0.8

    def __post_init__(self) -> None:
        """Normalise scalar fields and reject invalid configuration."""
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        for name in ("x", "theta", "k", "beta", "gamma", "theta_spike", "x_threshold"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        for name in ("x", "theta"):
            if not -5.0 <= getattr(self, name) <= 5.0:
                raise ValueError(f"{name} must be within [-5, 5]")
        if not 0.0 <= self.k <= 5.0:
            raise ValueError("k must be within [0, 5]")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be within [0, 1]")
        if not 0.0 <= self.gamma <= 2.0:
            raise ValueError("gamma must be within [0, 2]")
        for name in ("theta_spike", "x_threshold"):
            if not 0.0 <= getattr(self, name) <= 2.0:
                raise ValueError(f"{name} must be within [0, 2]")

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0.0:
            return 1.0 / (1.0 + math.exp(-value))
        exponential = math.exp(value)
        return exponential / (1.0 + exponential)

    def _candidate(self, current: float) -> tuple[float, float, int]:
        try:
            drive = float(current)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("current must be numeric") from exc
        if not math.isfinite(drive):
            raise ValueError("current must be finite")
        self._validate_configuration()
        previous_x = self.x
        activation = self._sigmoid((self.x - self.theta) * 4.0)
        next_x = -self.x + self.k * activation + drive
        fired = 1.0 if self.x >= self.theta_spike else 0.0
        next_theta = self.beta * self.theta + self.gamma * fired
        if not math.isfinite(next_x) or not math.isfinite(next_theta):
            raise FloatingPointError("SC adaptive-threshold candidate must be finite")
        bounded_x = max(-5.0, min(5.0, next_x))
        bounded_theta = max(-5.0, min(5.0, next_theta))
        event = int(bounded_x >= self.x_threshold and previous_x < self.x_threshold)
        return bounded_x, bounded_theta, event

    def step(self, current: float = 0.0) -> int:
        """Advance atomically and return an upward-threshold crossing event."""
        next_x, next_theta, event = self._candidate(current)
        self.x, self.theta = next_x, next_theta
        return event

    def simulate(
        self,
        current: npt.ArrayLike,
        *,
        backend: str = "auto",
    ) -> SCAdaptiveThresholdMapResult:
        """Run an atomic complete-state batch on a maintained backend."""
        from sc_neurocore.accel.sc_adaptive_threshold_map import (
            simulate_sc_adaptive_threshold_map,
        )

        result = simulate_sc_adaptive_threshold_map(
            self.x,
            self.theta,
            self.k,
            self.beta,
            self.gamma,
            self.theta_spike,
            self.x_threshold,
            current,
            backend=backend,
        )
        self.x = float(cast(float, result["x_final"]))
        self.theta = float(cast(float, result["theta_final"]))
        return result

    def reset(self) -> None:
        """Restore both project-model states while preserving parameters."""
        self.x = 0.0
        self.theta = 0.0


__all__ = ["SCAdaptiveThresholdMapNeuron", "SCAdaptiveThresholdMapResult"]
