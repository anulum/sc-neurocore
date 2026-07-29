# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — deterministic external drive for the SC Compte network

"""Counter-addressed Poisson input for ``SC-COMPTE-WM-NETWORK``.

The source paper describes aggregate 1,800 Hz independent Poisson input per
cell, but not a portable random-number stream.  This module therefore defines
an SC-owned, language-portable SplitMix64 counter mapping and exact inverse-CDF
Poisson sampling.  A sample is addressed only by seed, stream, step, and cell;
execution order and batching cannot change it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np

_UINT64_MAX = (1 << 64) - 1
_MASK64 = np.uint64(_UINT64_MAX)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_STEP_MIX = np.uint64(0xD1B54A32D192ED03)
_STREAM_MIX = np.uint64(0x94D049BB133111EB)


def _uint64(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _UINT64_MAX:
        raise ValueError(f"{name} must fit an unsigned 64-bit integer")
    return value


def _splitmix64(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return the fixed SplitMix64 finaliser over unsigned 64-bit counters."""
    with np.errstate(over="ignore"):
        z = values + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@dataclass(frozen=True, slots=True)
class CounterPoissonReceipt:
    """Auditable receipt for one population input sample."""

    step_index: int
    stream: int
    population_size: int
    total_events: int
    event_sha256: str


@dataclass(frozen=True, slots=True)
class CounterPoissonDrive:
    """Deterministic per-cell Poisson counts from a counter-addressed stream.

    ``rate_hz * dt_ms / 1000`` is the Poisson mean for one cell and timestep.
    The inverse CDF is built once through a residual tail below ``1e-15``.
    Counts are returned as signed 64-bit integers for portable FFI transport.
    """

    population_size: int
    rate_hz: float
    dt_ms: float
    seed: int
    stream: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.population_size, bool)
            or not isinstance(self.population_size, int)
            or self.population_size <= 0
        ):
            raise ValueError("population_size must be a positive integer")
        _uint64("seed", self.seed)
        _uint64("stream", self.stream)
        if not math.isfinite(self.rate_hz) or self.rate_hz < 0.0:
            raise ValueError("rate_hz must be finite and non-negative")
        if not math.isfinite(self.dt_ms) or self.dt_ms <= 0.0:
            raise ValueError("dt_ms must be finite and positive")
        if self.mean_events > 32.0:
            raise ValueError("counter Poisson mean exceeds the bounded SC input envelope")

    @property
    def mean_events(self) -> float:
        """Return the expected event count per cell and timestep."""
        return self.rate_hz * self.dt_ms / 1000.0

    def _cdf(self) -> np.ndarray[Any, Any]:
        mean = self.mean_events
        probability = math.exp(-mean)
        cumulative = probability
        values = [cumulative]
        count = 0
        while cumulative < 1.0 - 1.0e-15:
            count += 1
            if count > 255:
                raise ValueError("Poisson inverse CDF exceeded its bounded event range")
            probability *= mean / count
            cumulative += probability
            values.append(min(1.0, cumulative))
        values[-1] = 1.0
        return np.asarray(values, dtype=np.float64)

    def sample(self, step_index: int) -> tuple[np.ndarray[Any, Any], CounterPoissonReceipt]:
        """Return all cell counts and their canonical little-endian digest."""
        step = _uint64("step_index", step_index)
        cells = np.arange(self.population_size, dtype=np.uint64)
        with np.errstate(over="ignore"):
            counters = (
                np.uint64(self.seed)
                + np.uint64(step) * _STEP_MIX
                + np.uint64(self.stream) * _STREAM_MIX
                + cells * _GOLDEN
            ) & _MASK64
        random_bits = _splitmix64(counters)
        uniforms = ((random_bits >> np.uint64(11)).astype(np.float64) + 0.5) * (2.0**-53)
        counts = np.searchsorted(self._cdf(), uniforms, side="left").astype(np.int64)
        canonical = np.ascontiguousarray(counts, dtype="<i8")
        receipt = CounterPoissonReceipt(
            step_index=step,
            stream=self.stream,
            population_size=self.population_size,
            total_events=int(np.sum(counts, dtype=np.int64)),
            event_sha256=hashlib.sha256(canonical.tobytes()).hexdigest(),
        )
        return counts, receipt


__all__ = ["CounterPoissonDrive", "CounterPoissonReceipt"]
