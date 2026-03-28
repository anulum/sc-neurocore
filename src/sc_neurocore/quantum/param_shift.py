# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Gradient via parameter-shift rule

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def parameter_shift_gradient(
    circuit_fn: Callable[[np.ndarray[Any, Any]], float],
    params: np.ndarray[Any, Any],
    shift: float = float(np.pi / 2),
) -> np.ndarray[Any, Any]:
    """Gradient via parameter-shift rule.

    f'(θ_i) = [f(θ_i + s) - f(θ_i - s)] / (2 sin(s))
    """
    grad = np.zeros_like(params, dtype=float)
    denom = 2.0 * np.sin(shift)
    for i in range(len(params)):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += shift
        p_minus[i] -= shift
        grad[i] = (circuit_fn(p_plus) - circuit_fn(p_minus)) / denom
    return grad


class ParameterShiftOptimizer:
    def __init__(
        self,
        circuit_fn: Callable[[np.ndarray[Any, Any]], float],
        n_params: int,
        lr: float = 0.01,
    ) -> None:
        self.circuit_fn = circuit_fn
        self.n_params = n_params
        self.lr = lr

    def compute_gradient(self, params: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return parameter_shift_gradient(self.circuit_fn, params)

    def step(self, params: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        grad = self.compute_gradient(params)
        return params - self.lr * grad
