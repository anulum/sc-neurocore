# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_solvers_ode.py

from __future__ import annotations

import math
import time
import numpy as np
import pytest
from sc_neurocore.solvers import (
    EulerSolver,
    HeunSolver,
    RK4Solver,
    DormandPrinceSolver,
    ExponentialEuler,
    ExactLIFSolver,
    StormerVerlet,
    LeapfrogSolver,
    RosenbrockEuler,
    ImplicitEuler,
    TrapezoidalRule,
    get_solver,
)
def decay_ode(t, y):
    """dy/dt = -y. Solution: y(t) = y0 * exp(-t)."""
    return -y
def stiff_ode(t, y):
    """dy/dt = -1000*y. Very stiff decay."""
    return -1000.0 * y
def harmonic_oscillator(t, y):
    """Simple harmonic oscillator: dq/dt = p, dp/dt = -q.
    State: [q, p]. Conserves H = (q^2 + p^2) / 2.
    """
    return np.array([y[1], -y[0]])

__all__ = ['math', 'time', 'np', 'pytest', 'EulerSolver', 'HeunSolver', 'RK4Solver', 'DormandPrinceSolver', 'ExponentialEuler', 'ExactLIFSolver', 'StormerVerlet', 'LeapfrogSolver', 'RosenbrockEuler', 'ImplicitEuler', 'TrapezoidalRule', 'get_solver', 'decay_ode', 'stiff_ode', 'harmonic_oscillator', '__all__']
