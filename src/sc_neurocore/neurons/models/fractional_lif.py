# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fractional-order LIF — memory-dependent dynamics

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
    _max_history: int = 100

    def __post_init__(self) -> None:
        self._history: list[float] = [0.0] * self._max_history
        self._gl_coeffs: list[float] = self._compute_gl_coefficients()

    def _compute_gl_coefficients(self) -> list[float]:
        coeffs = [1.0]
        for k in range(1, self._max_history):
            coeffs.append(coeffs[-1] * (k - 1 - self.alpha) / k)
        return coeffs

    def step(self, current: float) -> int:
        rhs = -(self.v - self.v_rest) + self.resistance * current
        history = self._history
        gl_sum = sum(
            self._gl_coeffs[k] * history[-(k + 1)]
            for k in range(1, min(len(history), self._max_history))
            if len(history) > k
        )
        self.v = rhs * self.dt**self.alpha - gl_sum
        history.append(self.v)
        if len(history) > self._max_history:
            history.pop(0)

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            history[-1] = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self._history = [0.0] * self._max_history
