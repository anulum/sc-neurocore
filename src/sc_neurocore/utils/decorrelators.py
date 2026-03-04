from typing import Any, Optional
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .rng import RNG


@dataclass
class Decorrelator:
    """
    Base class for bitstream decorrelators.
    """

    def process(self, bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        raise NotImplementedError


@dataclass
class ShufflingDecorrelator(Decorrelator):
    """
    Decorrelates a bitstream by randomly shuffling bits within a window.
    This preserves the exact bit count (probability) but destroys temporal correlations.
    """

    window_size: int = 16
    seed: Optional[int] = None

    def __post_init__(self):  # type: ignore
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
            self._rng.shuffle(reshaped[i])  # type: ignore

        return reshaped.flatten()[:length]


@dataclass
class LFSRRegenDecorrelator(Decorrelator):
    """
    Regenerates a new bitstream with the same probability estimate
    but using a different random source (LFSR-like or just new RNG).
    """

    seed: Optional[int] = None

    def __post_init__(self):  # type: ignore
        self._rng = RNG(self.seed)

    def process(self, bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        p_est = bitstream.mean()
        # Regenerate
        return self._rng.bernoulli(p_est, size=len(bitstream)).astype(np.uint8)  # type: ignore
