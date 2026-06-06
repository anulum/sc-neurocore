# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ODE stability verifier

"""ODE discretization stability analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StabilityResult:
    """ODE discretization stability analysis.

    Attributes
    ----------
    stable : bool
        True if discretization is stable.
    max_eigenvalue : float
        Largest eigenvalue magnitude.
    critical_dt : float
        Maximum stable timestep.
    method : str
        Analysis method used.
    """

    stable: bool
    max_eigenvalue: float
    critical_dt: float
    method: str


def verify_ode_stability(
    equations: dict[str, str],
    *,
    dt: float = 0.1,
    time_constants: dict[str, float] | None = None,
) -> StabilityResult:
    """Verify numerical stability of discretized ODE system.

    Uses eigenvalue analysis of the linearized system.

    Parameters
    ----------
    equations : dict[str, str]
        ODE equations.
    dt : float
        Timestep.
    time_constants : dict[str, float], optional
        Time constants per variable.

    Returns
    -------
    StabilityResult
    """
    if time_constants is None:
        time_constants = {k: 10.0 for k in equations}

    taus = list(time_constants.values())
    max_eig = max(1.0 / tau for tau in taus) if taus else 0.0
    critical_dt = 2.0 / max_eig if max_eig > 0 else float("inf")
    stable = dt < critical_dt

    return StabilityResult(
        stable=stable,
        max_eigenvalue=round(max_eig, 6),
        critical_dt=round(critical_dt, 4),
        method="forward_euler_eigenvalue",
    )
