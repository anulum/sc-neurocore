# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
import numpy as np
from typing import Optional


class RNG:
    """
    Thin wrapper around NumPy RNG to keep a single interface.

    Later this can be extended to support:
    - hardware TRNGs
    - p-bit devices
    - reproducible streams per neuron
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def normal(self, mean: float = 0.0, std: float = 1.0, size=None):  # type: ignore
        return self._rng.normal(mean, std, size)

    def uniform(self, low: float = 0.0, high: float = 1.0, size=None):  # type: ignore
        return self._rng.uniform(low, high, size)

    def bernoulli(self, p: float, size=None):  # type: ignore
        return self._rng.random(size) < p

    def random(self, size=None):  # type: ignore
        return self._rng.random(size)

    def shuffle(self, x):  # type: ignore
        return self._rng.shuffle(x)
