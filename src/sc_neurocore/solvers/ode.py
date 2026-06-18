# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ODE Solver Suite

"""Industrial-grade ODE solvers for neuron simulation.

Provides fixed-step (Euler, Heun, RK4), adaptive (Dormand-Prince RK45),
and exponential integrators for linear ODEs (exact LIF integration).
"""

from __future__ import annotations

import abc
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

# State vectors are dense float64 arrays throughout the solver suite.
Vector = NDArray[np.float64]


class ODESolver(abc.ABC):
    """Base class for ODE solvers: dy/dt = f(t, y)."""

    @abc.abstractmethod
    def step(
        self,
        f: Callable[[float, Vector], Vector],
        y: Vector,
        t: float,
        dt: float,
    ) -> tuple[Vector, float]:
        """Advance one step. Returns (y_new, dt_used)."""


class EulerSolver(ODESolver):
    """Forward Euler — O(h).

    y_{n+1} = y_n + h * f(t_n, y_n)
    """

    def step(
        self,
        f: Callable[[float, Vector], Vector],
        y: Vector,
        t: float,
        dt: float,
    ) -> tuple[Vector, float]:
        """Advance one forward-Euler step; return ``(y_new, dt_used)``."""
        return y + dt * f(t, y), dt


class HeunSolver(ODESolver):
    """Heun's method (improved Euler / explicit trapezoidal) — O(h²).

    k1 = f(t_n, y_n)
    k2 = f(t_n + h, y_n + h*k1)
    y_{n+1} = y_n + h/2 * (k1 + k2)
    """

    def step(
        self,
        f: Callable[[float, Vector], Vector],
        y: Vector,
        t: float,
        dt: float,
    ) -> tuple[Vector, float]:
        """Advance one Heun (improved-Euler) step; return ``(y_new, dt_used)``."""
        k1 = f(t, y)
        k2 = f(t + dt, y + dt * k1)
        return y + 0.5 * dt * (k1 + k2), dt


class RK4Solver(ODESolver):
    """Classical Runge-Kutta — O(h⁴).

    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h, y + h * k3)
    y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    Reference: Kutta, W. (1901). Z. Math. Phys. 46:435–453.
    """

    def step(
        self,
        f: Callable[[float, Vector], Vector],
        y: Vector,
        t: float,
        dt: float,
    ) -> tuple[Vector, float]:
        """Advance one classical RK4 step; return ``(y_new, dt_used)``."""
        k1 = f(t, y)
        k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = f(t + dt, y + dt * k3)
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), dt


class DormandPrinceSolver(ODESolver):
    """Dormand-Prince adaptive RK45 — embedded pair for step-size control.

    Uses the 7-stage, 5th-order solution with a 4th-order embedded error
    estimate. Step size is adapted to maintain local truncation error
    below the specified tolerance.

    Reference: Dormand, J.R. & Prince, P.J. (1980). J. Comput. Appl. Math. 6:19–26.
    """

    # Butcher tableau coefficients
    _a2 = 1.0 / 5.0
    _a3 = 3.0 / 10.0
    _a4 = 4.0 / 5.0
    _a5 = 8.0 / 9.0

    _b21 = 1.0 / 5.0
    _b31, _b32 = 3.0 / 40.0, 9.0 / 40.0
    _b41, _b42, _b43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
    _b51, _b52, _b53, _b54 = 19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0
    _b61, _b62, _b63, _b64, _b65 = (
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
    )

    # 5th-order weights
    _c1, _c3, _c4, _c5, _c6 = (
        35.0 / 384.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    )

    # Error weights (5th - 4th)
    _e1 = 71.0 / 57600.0
    _e3 = -71.0 / 16695.0
    _e4 = 71.0 / 1920.0
    _e5 = -17253.0 / 339200.0
    _e6 = 22.0 / 525.0
    _e7 = -1.0 / 40.0

    def __init__(
        self,
        atol: float = 1e-8,
        rtol: float = 1e-6,
        max_factor: float = 5.0,
        min_factor: float = 0.2,
        safety: float = 0.9,
    ) -> None:
        self.atol = atol
        self.rtol = rtol
        self.max_factor = max_factor
        self.min_factor = min_factor
        self.safety = safety

    def step(  # type: ignore[override]
        self,
        f: Callable[[float, Vector], Vector],
        y: Vector,
        t: float,
        dt: float,
    ) -> tuple[Vector, float, float]:
        """Advance one adaptive RK45 step; return ``(y_new, dt_used, dt_next)``."""
        while True:
            k1 = f(t, y)
            k2 = f(t + self._a2 * dt, y + dt * self._b21 * k1)
            k3 = f(t + self._a3 * dt, y + dt * (self._b31 * k1 + self._b32 * k2))
            k4 = f(t + self._a4 * dt, y + dt * (self._b41 * k1 + self._b42 * k2 + self._b43 * k3))
            k5 = f(
                t + self._a5 * dt,
                y + dt * (self._b51 * k1 + self._b52 * k2 + self._b53 * k3 + self._b54 * k4),
            )
            k6 = f(
                t + dt,
                y
                + dt
                * (
                    self._b61 * k1
                    + self._b62 * k2
                    + self._b63 * k3
                    + self._b64 * k4
                    + self._b65 * k5
                ),
            )

            y5 = y + dt * (
                self._c1 * k1 + self._c3 * k3 + self._c4 * k4 + self._c5 * k5 + self._c6 * k6
            )

            # Error estimate
            k7 = f(t + dt, y5)
            err = dt * (
                self._e1 * k1
                + self._e3 * k3
                + self._e4 * k4
                + self._e5 * k5
                + self._e6 * k6
                + self._e7 * k7
            )

            scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y5))
            err_norm = np.sqrt(np.mean((err / scale) ** 2))

            if err_norm <= 1.0:
                if err_norm == 0.0:
                    factor = self.max_factor
                else:
                    factor = min(
                        self.max_factor, max(self.min_factor, self.safety * err_norm ** (-0.2))
                    )
                dt_next = dt * factor
                return y5, dt, dt_next
            else:
                factor = max(self.min_factor, self.safety * err_norm ** (-0.25))
                dt = dt * factor

    def integrate(
        self,
        f: Callable[[float, Vector], Vector],
        y0: Vector,
        t_span: tuple[float, float],
        dt0: float = 0.01,
    ) -> tuple[Vector, Vector]:
        """Integrate over [t0, tf]. Returns (t_array, y_array)."""
        t0, tf = t_span
        t = t0
        y = np.asarray(y0, dtype=float)
        dt = min(dt0, tf - t0)

        ts = [t]
        ys = [y.copy()]

        while t < tf:
            dt = min(dt, tf - t)
            y_new, dt_used, dt_next = self.step(f, y, t, dt)
            t += dt_used
            y = y_new
            dt = dt_next
            ts.append(t)
            ys.append(y.copy())

        return np.array(ts), np.array(ys)


class ExponentialEuler(ODESolver):
    """Exponential Euler for linear ODEs: dy/dt = A*y + b.

    Exact solution: y(t+dt) = exp(A*dt) * y(t) + (exp(A*dt) - I) * A^{-1} * b

    For scalar diagonal systems (like LIF):
    y(t+dt) = y_rest + (y(t) - y_rest) * exp(-dt/tau) + R*I * (1 - exp(-dt/tau))

    The callable f must return A*y + b; the solver extracts the decay constant.
    """

    def __init__(self, tau: float = 20.0, y_rest: float = -65.0, r_m: float = 1.0) -> None:
        self.tau = tau
        self.y_rest = y_rest
        self.r_m = r_m

    def step(
        self,
        f: Callable[[float, Vector], Vector],
        y: Vector,
        t: float,
        dt: float,
    ) -> tuple[Vector, float]:
        """Advance one exponential-Euler step; return ``(y_new, dt_used)``."""
        decay = np.exp(-dt / self.tau)
        # f is expected to return current only (scalar)
        current = f(t, y)
        # Same arithmetic shape either way: scalar or ndarray `current`
        # both broadcast against (y - self.y_rest) * decay.
        y_new = self.y_rest + (y - self.y_rest) * decay + self.r_m * current * (1.0 - decay)
        return y_new, dt


def get_solver(name: str, **kwargs: Any) -> ODESolver:
    """Return an ODE solver instance selected by name.

    Supported names: 'euler', 'heun', 'rk4', 'dp45', 'exponential_euler',
    'rosenbrock', and 'rosenbrock_euler'.
    """
    key = name.lower()
    if key == "euler":
        return EulerSolver()
    if key == "heun":
        return HeunSolver()
    if key == "rk4":
        return RK4Solver()
    if key == "dp45":
        return DormandPrinceSolver(**kwargs)
    if key == "exponential_euler":
        return ExponentialEuler(**kwargs)
    if key in {"rosenbrock", "rosenbrock_euler"}:
        from .stiff import RosenbrockEuler

        return RosenbrockEuler(**kwargs)
    raise ValueError(
        f"Unknown solver: {name!r}. Available: 'euler', 'heun', 'rk4', 'dp45', "
        "'exponential_euler', 'rosenbrock', 'rosenbrock_euler'."
    )
