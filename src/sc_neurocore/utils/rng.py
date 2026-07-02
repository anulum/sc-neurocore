# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reproducible NumPy random stream wrapper

"""Validated per-instance NumPy random streams for stochastic computing."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

DrawSize: TypeAlias = int | tuple[int, ...]
OptionalDrawSize: TypeAlias = DrawSize | None
FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]


def _validate_seed(seed: int | None) -> int | None:
    """Return a seed accepted by NumPy without implicit bool or negative aliases."""
    if seed is None:
        return None
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer or None")
    return seed


def _finite_float(value: float, name: str) -> float:
    """Coerce a scalar distribution parameter and reject non-finite values."""
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_std(std: float) -> float:
    """Return a strictly positive normal-distribution standard deviation."""
    result = _finite_float(std, "std")
    if result <= 0.0:
        raise ValueError("std must be positive")
    return result


def _uniform_bounds(low: float, high: float) -> tuple[float, float]:
    """Return finite, strictly ordered uniform-distribution bounds."""
    low_value = _finite_float(low, "low")
    high_value = _finite_float(high, "high")
    if low_value >= high_value:
        raise ValueError("low must be smaller than high")
    return low_value, high_value


def _probability(p: float) -> float:
    """Return a finite Bernoulli probability in the closed unit interval."""
    probability = _finite_float(p, "p")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("p must be in [0, 1]")
    return probability


class RNG:
    """Deterministic wrapper around NumPy's per-instance random generator.

    The wrapper is intentionally small: it exposes the distributions used by
    stochastic-computing utilities while enforcing scalar parameter domains
    before NumPy mutates generator state. Scalar draws return Python scalars;
    shaped draws return dtype-stable NumPy arrays.

    Example
    -------
    >>> rng = RNG(seed=42)
    >>> vals = rng.random(5)
    >>> vals.shape
    (5,)
    >>> RNG(seed=42).random(5) == vals  # deterministic
    array([ True,  True,  True,  True,  True])
    """

    def __init__(self, seed: int | None = None) -> None:
        """Create an independent stream from a non-negative integer seed.

        Parameters
        ----------
        seed:
            Optional non-negative seed. ``None`` delegates entropy selection to
            NumPy. Boolean values are rejected because they silently alias the
            integer seeds ``0`` and ``1``.

        Raises
        ------
        ValueError
            If ``seed`` is negative or boolean.
        """
        self._rng = np.random.default_rng(_validate_seed(seed))

    def normal(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        size: OptionalDrawSize = None,
    ) -> float | FloatArray:
        """Draw samples from a normal distribution.

        Parameters
        ----------
        mean:
            Finite distribution mean.
        std:
            Finite, strictly positive standard deviation.
        size:
            Optional output shape. ``None`` returns a Python ``float``.

        Returns
        -------
        float | numpy.ndarray
            A scalar float for scalar draws, otherwise a ``float64`` array.

        Raises
        ------
        ValueError
            If ``mean`` is not finite or ``std`` is not finite and positive.
        """
        mean_value = _finite_float(mean, "mean")
        std_value = _positive_std(std)
        samples = self._rng.normal(mean_value, std_value, size)
        if size is None:
            return float(samples)
        return np.asarray(samples, dtype=np.float64)

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: OptionalDrawSize = None,
    ) -> float | FloatArray:
        """Draw samples from a bounded uniform distribution.

        Parameters
        ----------
        low:
            Finite inclusive lower bound.
        high:
            Finite exclusive upper bound. It must be greater than ``low``.
        size:
            Optional output shape. ``None`` returns a Python ``float``.

        Returns
        -------
        float | numpy.ndarray
            A scalar float for scalar draws, otherwise a ``float64`` array.

        Raises
        ------
        ValueError
            If bounds are not finite or are not strictly ordered.
        """
        low_value, high_value = _uniform_bounds(low, high)
        samples = self._rng.uniform(low_value, high_value, size)
        if size is None:
            return float(samples)
        return np.asarray(samples, dtype=np.float64)

    def bernoulli(self, p: float, size: OptionalDrawSize = None) -> bool | BoolArray:
        """Draw Bernoulli samples from a validated probability.

        Parameters
        ----------
        p:
            Finite probability in the closed interval ``[0, 1]``.
        size:
            Optional output shape. ``None`` returns a Python ``bool``.

        Returns
        -------
        bool | numpy.ndarray
            A scalar boolean for scalar draws, otherwise a ``bool`` array.

        Raises
        ------
        ValueError
            If ``p`` is non-finite or outside ``[0, 1]``.
        """
        probability = _probability(p)
        samples = self._rng.random(size) < probability
        if size is None:
            return bool(samples)
        return np.asarray(samples, dtype=np.bool_)

    def random(self, size: OptionalDrawSize = None) -> float | FloatArray:
        """Draw uniform samples from the half-open interval ``[0, 1)``.

        Parameters
        ----------
        size:
            Optional output shape. ``None`` returns a Python ``float``.

        Returns
        -------
        float | numpy.ndarray
            A scalar float for scalar draws, otherwise a ``float64`` array.
        """
        samples = self._rng.random(size)
        if size is None:
            return float(samples)
        return np.asarray(samples, dtype=np.float64)

    def shuffle(self, x: npt.NDArray[np.generic]) -> None:
        """Shuffle an array in place using this instance's random stream.

        Parameters
        ----------
        x:
            Mutable NumPy array whose first axis is permuted in place.
        """
        self._rng.shuffle(x)
