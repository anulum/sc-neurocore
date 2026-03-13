# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
import numpy as np
from typing import Optional


class RNG:
    """
    Thin wrapper around NumPy RNG for reproducible per-neuron streams.

    Example
    -------
    >>> rng = RNG(seed=42)
    >>> vals = rng.random(5)
    >>> vals.shape
    (5,)
    >>> RNG(seed=42).random(5) == vals  # deterministic
    array([ True,  True,  True,  True,  True])
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def normal(self, mean: float = 0.0, std: float = 1.0, size=None):
        return self._rng.normal(mean, std, size)

    def uniform(self, low: float = 0.0, high: float = 1.0, size=None):
        return self._rng.uniform(low, high, size)

    def bernoulli(self, p: float, size=None):
        return self._rng.random(size) < p

    def random(self, size=None):
        return self._rng.random(size)

    def shuffle(self, x):
        return self._rng.shuffle(x)
