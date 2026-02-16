import numpy as np
from dataclasses import dataclass

from ..accel._dispatch import njit_or_python


@njit_or_python(cache=True)
def _logistic_map(r: float, x0: float, size: int) -> tuple:  # pragma: no cover
    """JIT logistic map: returns (output_array, final_state)."""
    out = np.zeros(size)
    curr = x0
    for i in range(size):
        curr = r * curr * (1.0 - curr)
        out[i] = curr
    return out, curr


@dataclass
class ChaoticRNG:
    """
    Chaotic Random Number Generator using Logistic Map.
    x_{n+1} = r * x_n * (1 - x_n)

    Provides 'True' Randomness simulation (Deterministic Chaos)
    unlike linear PRNGs.
    """

    r: float = 4.0
    x: float = 0.5

    def __post_init__(self):
        # Burn-in to forget initial state
        _, self.x = _logistic_map(self.r, self.x, 100)

    def random(self, size: int) -> np.ndarray:
        """Generate 'size' random floats [0, 1]."""
        out, self.x = _logistic_map(self.r, self.x, size)
        return out

    def generate_bitstream(self, p: float, length: int) -> np.ndarray:
        """Generate bitstream using chaotic source."""
        vals = self.random(length)
        return (vals < p).astype(np.uint8)
