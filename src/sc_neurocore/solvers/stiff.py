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


import numpy as np

from typing import Callable

from numpy.typing import NDArray

from .ode import ODESolver


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
        f: Callable[[float, NDArray], NDArray],
        y: NDArray,
        t: float,
        dt: float,
    ) -> tuple[NDArray, float]:
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
        f: Callable[[float, NDArray], NDArray],
        y: NDArray,
        t: float,
        dt: float,
    ) -> tuple[NDArray, float]:
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
