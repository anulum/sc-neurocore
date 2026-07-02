# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Base class for bitstream decorrelators

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .rng import RNG


@dataclass
class Decorrelator(ABC):
    """
    Base class for bitstream decorrelators.
    """

    @abstractmethod
    def process(self, bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]: ...


@dataclass
class ShufflingDecorrelator(Decorrelator):
    """
    Decorrelates a bitstream by randomly shuffling bits within a window.
    This preserves the exact bit count (probability) but destroys temporal correlations.
    """

    window_size: int = 16
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = RNG(self.seed)

    def process(self, bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        # Reshape into windows
        length = len(bitstream)
        pad = (self.window_size - (length % self.window_size)) % self.window_size

        if pad > 0:
            padded = np.append(bitstream, np.zeros(pad, dtype=np.uint8))
        else:
            padded = bitstream.copy()

        num_windows = len(padded) // self.window_size
        reshaped = padded.reshape((num_windows, self.window_size))

        # Shuffle each row
        # Note: Ideally we want independent shuffles per row.
        # fast way:
        for i in range(num_windows):
            self._rng.shuffle(reshaped[i])

        return reshaped.flatten()[:length]


@dataclass
class LFSRRegenDecorrelator(Decorrelator):
    """
    Regenerates a new bitstream with the same probability estimate
    but using a different random source (LFSR-like or just new RNG).
    """

    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = RNG(self.seed)

    def process(self, bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        p_est = bitstream.mean()
        # Regenerate. ``bernoulli`` is typed ``bool | ndarray`` (scalar for a
        # scalar draw); ``np.asarray`` narrows the sized draw to an array before
        # the dtype cast so the ``.astype`` union-attr does not reach mypy.
        regenerated: np.ndarray[Any, Any] = np.asarray(
            self._rng.bernoulli(p_est, size=len(bitstream)), dtype=np.uint8
        )
        return regenerated
