# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Symplectic ODE Solvers for Hamiltonian Systems

"""Symplectic integrators that preserve phase-space volume.

Essential for long-time simulation of oscillatory neuron models
(FitzHugh-Nagumo, resonate-and-fire, theta neuron) where energy
conservation matters.
"""

from __future__ import annotations


import numpy as np

from typing import Callable

from numpy.typing import NDArray

from .ode import ODESolver


def _validate_symplectic_inputs(y: NDArray, t: float, dt: float) -> NDArray:
    if isinstance(dt, bool) or not isinstance(dt, int | float):
        raise ValueError("dt must be a finite positive number")
    if not np.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be a finite positive number")
    if isinstance(t, bool) or not isinstance(t, int | float):
        raise ValueError("t must be finite")
    if not np.isfinite(float(t)):
        raise ValueError("t must be finite")
    state = np.asarray(y, dtype=np.float64)
    if state.ndim != 1 or state.size == 0 or state.size % 2 != 0:
        raise ValueError("symplectic state must be a non-empty even-length 1-D array")
    if not np.all(np.isfinite(state)):
        raise ValueError("symplectic state must contain only finite values")
    return state


def _validated_rhs(
    f: Callable[[float, NDArray], NDArray],
    t: float,
    y: NDArray,
) -> NDArray:
    dy = np.asarray(f(t, y), dtype=np.float64)
    if dy.shape != y.shape:
        raise ValueError("symplectic RHS must return an array matching the state shape")
    if not np.all(np.isfinite(dy)):
        raise ValueError("symplectic RHS must contain only finite values")
    return dy


class StormerVerlet(ODESolver):
    """Störmer-Verlet (velocity Verlet) — symplectic 2nd order.

    For separable Hamiltonians H = T(p) + V(q):
        p_{1/2} = p_n - (h/2) * ∇V(q_n)
        q_{n+1} = q_n + h * ∇T(p_{1/2})
        p_{n+1} = p_{1/2} - (h/2) * ∇V(q_{n+1})

    Here we split the state y = [q, p] where q are position-like (voltage)
    and p are momentum-like (recovery/gating) variables.

    Reference: Hairer, E. et al. (2006). Geometric Numerical Integration. Springer.
    """

    def step(
        self,
        f: Callable[[float, NDArray], NDArray],
        y: NDArray,
        t: float,
        dt: float,
    ) -> tuple[NDArray, float]:
        y = _validate_symplectic_inputs(y, t, dt)
        dt = float(dt)
        n = len(y) // 2
        q, p = y[:n], y[n:]

        # Full evaluation to get derivatives
        dy = _validated_rhs(f, float(t), y)
        dq, dp = dy[:n], dy[n:]

        # Half kick
        p_half = p + 0.5 * dt * dp
        # Drift
        y_temp = np.concatenate([q, p_half])
        dy_temp = _validated_rhs(f, float(t) + 0.5 * dt, y_temp)
        with np.errstate(over="ignore", invalid="ignore"):
            q_new = q + dt * dy_temp[:n]
        # Half kick
        y_temp2 = np.concatenate([q_new, p_half])
        dy_final = _validated_rhs(f, float(t) + dt, y_temp2)
        with np.errstate(over="ignore", invalid="ignore"):
            p_new = p_half + 0.5 * dt * dy_final[n:]

        y_new = np.concatenate([q_new, p_new])
        if not np.all(np.isfinite(y_new)):
            raise ValueError("symplectic update produced non-finite state")
        return y_new, dt


class LeapfrogSolver(ODESolver):
    """Leapfrog (kick-drift-kick) — symplectic 2nd order.

    Equivalent to Störmer-Verlet but staggered:
        p_{n+1/2} = p_n + (h/2) * f_p(q_n)
        q_{n+1} = q_n + h * f_q(p_{n+1/2})
        p_{n+1} = p_{n+1/2} + (h/2) * f_p(q_{n+1})

    State vector: y = [q₀..q_{n-1}, p₀..p_{n-1}]

    Reference: Yoshida, H. (1990). Phys. Lett. A 150:262–268.
    """

    def step(
        self,
        f: Callable[[float, NDArray], NDArray],
        y: NDArray,
        t: float,
        dt: float,
    ) -> tuple[NDArray, float]:
        y = _validate_symplectic_inputs(y, t, dt)
        dt = float(dt)
        n = len(y) // 2
        q, p = y[:n].copy(), y[n:].copy()

        dy0 = _validated_rhs(f, float(t), y)
        dp0 = dy0[n:]

        p_half = p + 0.5 * dt * dp0

        y_mid = np.concatenate([q, p_half])
        dy_mid = _validated_rhs(f, float(t), y_mid)
        dq_mid = dy_mid[:n]
        with np.errstate(over="ignore", invalid="ignore"):
            q_new = q + dt * dq_mid

        y_end = np.concatenate([q_new, p_half])
        dy_end = _validated_rhs(f, float(t) + dt, y_end)
        dp_end = dy_end[n:]
        with np.errstate(over="ignore", invalid="ignore"):
            p_new = p_half + 0.5 * dt * dp_end

        y_new = np.concatenate([q_new, p_new])
        if not np.all(np.isfinite(y_new)):
            raise ValueError("symplectic update produced non-finite state")
        return y_new, dt
