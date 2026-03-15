# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FractionalLIFNeuron:
    """Fractional-order LIF — memory-dependent dynamics.

    Uses Grünwald-Letnikov fractional derivative approximation.
    D^α v(t) = -(v - v_rest) + R·I, where 0 < α ≤ 1.
    α < 1 introduces memory (power-law decay instead of exponential).
    Lundstrom et al. 2008.
    """

    v: float = 0.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_threshold: float = 1.0
    alpha: float = 0.8
    resistance: float = 1.0
    dt: float = 1.0
    _history: list = None
    _max_history: int = 100

    def __post_init__(self):
        self._history = [0.0] * self._max_history
        self._gl_coeffs = self._compute_gl_coefficients()

    def _compute_gl_coefficients(self):
        coeffs = [1.0]
        for k in range(1, self._max_history):
            coeffs.append(coeffs[-1] * (k - 1 - self.alpha) / k)
        return coeffs

    def step(self, current: float) -> int:
        rhs = -(self.v - self.v_rest) + self.resistance * current
        gl_sum = sum(
            self._gl_coeffs[k] * self._history[-(k + 1)]
            for k in range(1, min(len(self._history), self._max_history))
            if len(self._history) > k
        )
        self.v = rhs * self.dt**self.alpha - gl_sum
        self._history.append(self.v)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self._history[-1] = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self._history = [0.0] * self._max_history
