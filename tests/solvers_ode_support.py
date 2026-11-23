# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_solvers_ode.py

"""Shared reference right-hand sides for the ODE solver suites."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def decay_ode(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Decay ODE dy/dt = -y, whose solution is y(t) = y0 * exp(-t)."""
    return -y


def stiff_ode(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Very stiff decay ODE dy/dt = -1000*y."""
    return -1000.0 * y


def harmonic_oscillator(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the simple harmonic oscillator field dq/dt = p, dp/dt = -q.

    State: [q, p]. Conserves H = (q^2 + p^2) / 2.
    """
    return np.array([y[1], -y[0]])


__all__ = [
    "decay_ode",
    "stiff_ode",
    "harmonic_oscillator",
]
