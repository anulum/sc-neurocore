# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.solvers -- Tier: core (production)

"""sc_neurocore.solvers — ODE integration and combinatorial solvers.

Includes:
- Fixed-step ODE solvers: Euler, Heun, RK4
- Adaptive ODE solver: Dormand-Prince RK45
- Exponential integrator for linear ODEs
- Exact event-driven LIF solver
- Symplectic integrators: Störmer-Verlet, Leapfrog
- Implicit solvers: Rosenbrock-Euler, Backward Euler, Trapezoidal
  (Crank-Nicolson)
- Combinatorial: Stochastic Ising Graph
"""

__tier__ = "core"

from .ising import StochasticIsingGraph
from .ode import (
    ODESolver,
    EulerSolver,
    HeunSolver,
    RK4Solver,
    DormandPrinceSolver,
    ExponentialEuler,
    get_solver,
)
from .exact_lif import ExactLIFSolver
from .exact_lif_profile import (
    CurrentDriveTick,
    ExactCurrentLIFProfile,
    ExactCurrentLIFSession,
    ExactLIFEvent,
    ExactLIFExecutionPacket,
    ExactLIFState,
    ExactLIFStateSample,
)
from .symplectic import StormerVerlet, LeapfrogSolver
from .stiff import ImplicitEuler, RosenbrockEuler, TrapezoidalRule

__all__ = [
    "StochasticIsingGraph",
    "ODESolver",
    "EulerSolver",
    "HeunSolver",
    "RK4Solver",
    "DormandPrinceSolver",
    "ExponentialEuler",
    "ExactLIFSolver",
    "CurrentDriveTick",
    "ExactCurrentLIFProfile",
    "ExactCurrentLIFSession",
    "ExactLIFEvent",
    "ExactLIFExecutionPacket",
    "ExactLIFState",
    "ExactLIFStateSample",
    "StormerVerlet",
    "LeapfrogSolver",
    "RosenbrockEuler",
    "ImplicitEuler",
    "TrapezoidalRule",
    "get_solver",
]
