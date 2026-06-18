# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stimulus sources: timed arrays, Poisson input, step currents

"""Stimulus sources for network simulations."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .population import Population


class TimedArray:
    """Time-varying current from a pre-computed array."""

    def __init__(self, values: np.ndarray[Any, Any] | list[float], dt: float = 0.001) -> None:
        self.values = np.asarray(values, dtype=np.float64)
        self.dt = dt
        self.target: Population | None = None

    def get_current(self, t_step: int) -> float:
        """Return the value at timestep t_step (clamps to last value)."""
        idx = min(t_step, len(self.values) - 1)
        return float(self.values[idx])


class PoissonInput:
    """Random Poisson spike input producing weighted current."""

    def __init__(
        self, n: int, rate_hz: float, weight: float, dt: float = 0.001, seed: int = 42
    ) -> None:
        self.n = n
        self.rate_hz = rate_hz
        self.weight = weight
        self.dt = dt
        self._rng = np.random.default_rng(seed)
        self.target: Population | None = None

    def get_current(self, t_step: int, dt: float | None = None) -> np.ndarray[Any, Any]:
        """Generate Poisson spikes and return weighted current vector."""
        step_dt = dt if dt is not None else self.dt
        p_spike = self.rate_hz * step_dt
        spikes = (self._rng.random(self.n) < p_spike).astype(np.float64)
        return spikes * self.weight


class StepCurrent:
    """Rectangular step current between onset and offset timesteps."""

    def __init__(self, onset: int, offset: int, amplitude: float) -> None:
        self.onset = onset
        self.offset = offset
        self.amplitude = amplitude
        self.target: Population | None = None

    def get_current(self, t_step: int, dt: float = 0.001) -> float:
        """Return amplitude if within [onset, offset), else 0."""
        if self.onset <= t_step < self.offset:
            return self.amplitude
        return 0.0
