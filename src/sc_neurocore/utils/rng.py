# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Thin wrapper around NumPy RNG for reproducible

from __future__ import annotations

from typing import Any, Optional

import numpy as np


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

    def normal(
        self, mean: float = 0.0, std: float = 1.0, size: int | tuple[int, ...] | None = None
    ) -> Any:
        return self._rng.normal(mean, std, size)

    def uniform(
        self, low: float = 0.0, high: float = 1.0, size: int | tuple[int, ...] | None = None
    ) -> Any:
        return self._rng.uniform(low, high, size)

    def bernoulli(self, p: float, size: int | tuple[int, ...] | None = None) -> Any:
        return self._rng.random(size) < p

    def random(self, size: int | tuple[int, ...] | None = None) -> Any:
        return self._rng.random(size)

    def shuffle(self, x: np.ndarray[Any, Any]) -> None:
        self._rng.shuffle(x)
