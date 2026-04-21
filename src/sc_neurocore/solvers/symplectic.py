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
        n = len(y) // 2
        q, p = y[:n], y[n:]

        # Full evaluation to get derivatives
        dy = f(t, y)
        dq, dp = dy[:n], dy[n:]

        # Half kick
        p_half = p + 0.5 * dt * dp
        # Drift
        y_temp = np.concatenate([q, p_half])
        dy_temp = f(t + 0.5 * dt, y_temp)
        q_new = q + dt * dy_temp[:n]
        # Half kick
        y_temp2 = np.concatenate([q_new, p_half])
        dy_final = f(t + dt, y_temp2)
        p_new = p_half + 0.5 * dt * dy_final[n:]

        return np.concatenate([q_new, p_new]), dt


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
        n = len(y) // 2
        q, p = y[:n].copy(), y[n:].copy()

        dy0 = f(t, y)
        dp0 = dy0[n:]

        p_half = p + 0.5 * dt * dp0

        y_mid = np.concatenate([q, p_half])
        dy_mid = f(t, y_mid)
        dq_mid = dy_mid[:n]
        q_new = q + dt * dq_mid

        y_end = np.concatenate([q_new, p_half])
        dy_end = f(t + dt, y_end)
        dp_end = dy_end[n:]
        p_new = p_half + 0.5 * dt * dp_end

        return np.concatenate([q_new, p_new]), dt
