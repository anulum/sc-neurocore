# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rall 1962 — N-compartment passive cable discretisation

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _finite_scalar(name: str, value: float) -> float:
    """Return a finite scalar or raise a descriptive validation error."""
    scalar = float(value)
    if not np.isfinite(scalar):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return scalar


def _solve_tridiagonal(
    lower: NDArray[np.float64],
    diagonal: NDArray[np.float64],
    upper: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve a tridiagonal system using the Thomas algorithm."""
    n = int(diagonal.size)
    if n == 0:
        msg = "tridiagonal system must contain at least one row"
        raise ValueError(msg)
    c_prime = np.zeros(max(n - 1, 0), dtype=np.float64)
    d_prime = np.zeros(n, dtype=np.float64)
    pivot = diagonal[0]
    if not np.isfinite(pivot) or pivot == 0.0:
        msg = "singular passive cable system"
        raise ValueError(msg)
    if n > 1:
        c_prime[0] = upper[0] / pivot
    d_prime[0] = rhs[0] / pivot
    for i in range(1, n):
        pivot = diagonal[i] - lower[i - 1] * c_prime[i - 1]
        if not np.isfinite(pivot) or pivot == 0.0:
            msg = "singular passive cable system"
            raise ValueError(msg)
        if i < n - 1:
            c_prime[i] = upper[i] / pivot
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / pivot
    solution = np.zeros(n, dtype=np.float64)
    solution[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]
    if not np.all(np.isfinite(solution)):
        msg = "passive cable solve produced non-finite state"
        raise ValueError(msg)
    return solution


@dataclass
class RallCableNeuron:
    """Rall 1962 N-compartment passive cable with an implicit step.

    The step solves the sealed-end passive cable operator as a tridiagonal
    backward-Euler system with distal current held constant over ``dt``.
    State is committed only after the finite candidate solve succeeds.

    Reference: Rall, W. (1959). Exp. Neurol. 1:491–527.
    """

    n_comp: int = 5
    tau_m: float = 20.0  # ms
    v_rest: float = -65.0  # mV
    g_ratio: float = 0.5  # g_axial / g_leak ratio (electrotonic coupling)
    v_threshold: float = -50.0
    v_reset: float = -65.0
    dt: float = 0.1
    v: np.ndarray[Any, Any] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.n_comp, int) or self.n_comp < 1:
            msg = "n_comp must be a positive integer"
            raise ValueError(msg)
        self.tau_m = _finite_scalar("tau_m", self.tau_m)
        self.v_rest = _finite_scalar("v_rest", self.v_rest)
        self.g_ratio = _finite_scalar("g_ratio", self.g_ratio)
        self.v_threshold = _finite_scalar("v_threshold", self.v_threshold)
        self.v_reset = _finite_scalar("v_reset", self.v_reset)
        self.dt = _finite_scalar("dt", self.dt)
        if self.tau_m <= 0.0:
            msg = "tau_m must be positive"
            raise ValueError(msg)
        if self.g_ratio < 0.0:
            msg = "g_ratio must be non-negative"
            raise ValueError(msg)
        if self.dt <= 0.0:
            msg = "dt must be positive"
            raise ValueError(msg)
        self.v = np.full(self.n_comp, self.v_rest, dtype=np.float64)

    def _implicit_candidate(self, current: float) -> NDArray[np.float64]:
        """Return the finite implicit passive-cable candidate voltage vector."""
        drive = _finite_scalar("current", current)
        if self.v.shape != (self.n_comp,) or not np.all(np.isfinite(self.v)):
            msg = "runtime cable state is corrupt"
            raise ValueError(msg)

        alpha = self.dt / self.tau_m
        offdiag = -alpha * self.g_ratio
        diagonal = np.full(self.n_comp, 1.0 + alpha + 2.0 * alpha * self.g_ratio, dtype=np.float64)
        if self.n_comp == 1:
            diagonal[0] = 1.0 + alpha
        else:
            diagonal[0] = 1.0 + alpha + alpha * self.g_ratio
            diagonal[-1] = 1.0 + alpha + alpha * self.g_ratio
        lower = np.full(max(self.n_comp - 1, 0), offdiag, dtype=np.float64)
        upper = np.full(max(self.n_comp - 1, 0), offdiag, dtype=np.float64)
        rhs = self.v - self.v_rest
        rhs = rhs.astype(np.float64, copy=True)
        rhs[-1] += alpha * drive
        return _solve_tridiagonal(lower, diagonal, upper, rhs) + self.v_rest

    def step(self, current: float) -> int:
        """Advance one implicit cable step and return the somatic spike flag."""
        previous_soma = float(self.v[0])
        candidate = self._implicit_candidate(current)
        if candidate[0] >= self.v_threshold and previous_soma < self.v_threshold:
            candidate[0] = self.v_reset
            self.v = candidate
            return 1
        self.v = candidate
        return 0

    def reset(self) -> None:
        """Reset all compartments to the leak reversal potential."""
        self.v[:] = self.v_rest
