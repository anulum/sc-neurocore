# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import numpy as np


def parameter_shift_gradient(circuit_fn, params, shift=np.pi / 2):
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
    def __init__(self, circuit_fn, n_params, lr=0.01):
        self.circuit_fn = circuit_fn
        self.n_params = n_params
        self.lr = lr

    def compute_gradient(self, params):
        return parameter_shift_gradient(self.circuit_fn, params)

    def step(self, params):
        grad = self.compute_gradient(params)
        return params - self.lr * grad
