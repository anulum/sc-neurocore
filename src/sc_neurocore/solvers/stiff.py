# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Implicit ODE Solvers for Stiff Systems

"""A-stable and L-stable implicit solvers for stiff Hodgkin-Huxley-type ODEs.

HH-type gating variables have time constants spanning 0.1ms to 100ms,
producing stiffness ratios up to 1000:1. Explicit methods require tiny
steps; implicit methods remain stable with larger steps.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .ode import ODESolver, Vector


def _finite_difference_jacobian(
    f: Callable[[float, Vector], Vector],
    t: float,
    y: Vector,
    f_y: Vector,
    epsilon: float,
) -> Vector:
    n = y.size
    jacobian = np.empty((n, n), dtype=np.float64)
    for col in range(n):
        delta = epsilon * max(1.0, abs(float(y[col])))
        shifted = y.copy()
        shifted[col] += delta
        jacobian[:, col] = (f(t, shifted) - f_y) / delta
    return jacobian


class RosenbrockEuler(ODESolver):
    """Linearly implicit one-stage Rosenbrock solver.

    The step solves ``(I - gamma*h*J) k = h*f(t, y)`` and returns
    ``y + k``. With ``gamma=1`` this is the Rosenbrock-Euler method:
    first-order, L-stable for linear stiff decay, and suitable as a
    stiffness-specific alternative path for small neuron state vectors.

    Reference: Hairer, E. & Wanner, G. (1996). Solving ODEs II. Springer.
    """

    def __init__(self, gamma: float = 1.0, jacobian_epsilon: float = 1e-6) -> None:
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if jacobian_epsilon <= 0.0:
            raise ValueError("jacobian_epsilon must be positive")
        self.gamma = gamma
        self.jacobian_epsilon = jacobian_epsilon

    def step(
        self,
        f: Callable[[float, Vector], Vector],
        y: Vector,
        t: float,
        dt: float,
    ) -> tuple[Vector, float]:
        """Advance one Rosenbrock-Euler step; return ``(y_new, dt_used)``."""
        y_vec = np.asarray(y, dtype=np.float64)
        f_y = np.asarray(f(t, y_vec), dtype=np.float64)
        jacobian = _finite_difference_jacobian(f, t, y_vec, f_y, self.jacobian_epsilon)
        system = np.eye(y_vec.size, dtype=np.float64) - self.gamma * dt * jacobian
        rhs = dt * f_y
        try:
            increment = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            increment = np.linalg.lstsq(system, rhs, rcond=None)[0]
        return y_vec + increment, dt


class ImplicitEuler(ODESolver):
    """Backward (implicit) Euler — L-stable, 1st order.

    y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})

    Solved via fixed-point iteration: for stiff neuron ODEs,
    3–5 iterations typically suffice.

    Reference: Hairer, E. & Wanner, G. (1996). Solving ODEs II. Springer.
    """

    def __init__(self, max_iterations: int = 10, tol: float = 1e-10) -> None:
        self.max_iterations = max_iterations
        self.tol = tol

    def step(
        self,
        f: Callable[[float, Vector], Vector],
        y: Vector,
        t: float,
        dt: float,
    ) -> tuple[Vector, float]:
        """Advance one backward-Euler step; return ``(y_new, dt_used)``."""
        t_next = t + dt
        y_next = y + dt * f(t, y)  # initial guess from forward Euler

        for _ in range(self.max_iterations):
            y_new = y + dt * f(t_next, y_next)
            if np.max(np.abs(y_new - y_next)) < self.tol:
                break
            y_next = y_new

        return y_next, dt


class TrapezoidalRule(ODESolver):
    """Trapezoidal rule (Crank-Nicolson) — A-stable, 2nd order.

    y_{n+1} = y_n + h/2 * (f(t_n, y_n) + f(t_{n+1}, y_{n+1}))

    Higher-order than implicit Euler while retaining A-stability.
    Solved via fixed-point iteration.

    Reference: Crank, J. & Nicolson, P. (1947). Proc. Camb. Phil. Soc. 43:50–67.
    """

    def __init__(self, max_iterations: int = 10, tol: float = 1e-10) -> None:
        self.max_iterations = max_iterations
        self.tol = tol

    def step(
        self,
        f: Callable[[float, Vector], Vector],
        y: Vector,
        t: float,
        dt: float,
    ) -> tuple[Vector, float]:
        """Advance one trapezoidal-rule step; return ``(y_new, dt_used)``."""
        f_n = f(t, y)
        t_next = t + dt
        y_next = y + dt * f_n  # initial guess

        for _ in range(self.max_iterations):
            f_next = f(t_next, y_next)
            y_new = y + 0.5 * dt * (f_n + f_next)
            if np.max(np.abs(y_new - y_next)) < self.tol:
                break
            y_next = y_new

        return y_next, dt
