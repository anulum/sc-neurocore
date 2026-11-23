# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSymplecticSolvers from former test_solvers_ode.py

"""Focused suite: TestSymplecticSolvers from former test_solvers_ode.py."""

from __future__ import annotations

from numpy.typing import NDArray
import numpy as np
import pytest
from sc_neurocore.solvers import (
    LeapfrogSolver,
    StormerVerlet,
)
from tests.solvers_ode_support import (
    harmonic_oscillator,
)


class TestSymplecticSolvers:
    """Energy-conserving symplectic solvers."""

    def _run_oscillator(
        self, solver: StormerVerlet | LeapfrogSolver, n_steps: int = 10000, dt: float = 0.01
    ) -> list[float]:
        y = np.array([1.0, 0.0])  # q=1, p=0
        energies: list[float] = []
        for _ in range(n_steps):
            y, _ = solver.step(harmonic_oscillator, y, 0.0, dt)
            energies.append(0.5 * (y[0] ** 2 + y[1] ** 2))
        return energies

    def test_verlet_energy_conservation(self) -> None:
        """Stormer-Verlet conserves oscillator energy."""
        energies = self._run_oscillator(StormerVerlet(), n_steps=10000)
        assert abs(energies[0] - energies[-1]) / energies[0] < 0.01

    def test_leapfrog_energy_conservation(self) -> None:
        """Leapfrog conserves oscillator energy."""
        energies = self._run_oscillator(LeapfrogSolver(), n_steps=10000)
        assert abs(energies[0] - energies[-1]) / energies[0] < 0.01

    @pytest.mark.parametrize("solver", [StormerVerlet(), LeapfrogSolver()])
    @pytest.mark.parametrize(
        ("y", "t", "dt", "match"),
        [
            (np.array([1.0]), 0.0, 0.01, "even-length"),
            (np.array([[1.0, 0.0]]), 0.0, 0.01, "1-D"),
            (np.array([1.0, np.nan]), 0.0, 0.01, "finite"),
            (np.array([1.0, 0.0]), True, 0.01, "t"),
            (np.array([1.0, 0.0]), float("nan"), 0.01, "t"),
            (np.array([1.0, 0.0]), 0.0, True, "dt"),
            (np.array([1.0, 0.0]), 0.0, 0.0, "dt"),
            (np.array([1.0, 0.0]), 0.0, float("inf"), "dt"),
        ],
    )
    def test_symplectic_solvers_reject_invalid_state_contracts(
        self,
        solver: StormerVerlet | LeapfrogSolver,
        y: NDArray[np.float64],
        t: float,
        dt: float,
        match: str,
    ) -> None:
        """Each invalid state, time or step is rejected."""
        with pytest.raises(ValueError, match=match):
            solver.step(harmonic_oscillator, y, t, dt)

    @pytest.mark.parametrize("solver", [StormerVerlet(), LeapfrogSolver()])
    def test_symplectic_solvers_reject_bad_rhs_shape(
        self, solver: StormerVerlet | LeapfrogSolver
    ) -> None:
        """A right-hand side of the wrong shape is rejected."""
        with pytest.raises(ValueError, match="state shape"):
            solver.step(lambda _t, _y: np.array([1.0]), np.array([1.0, 0.0]), 0.0, 0.01)

    @pytest.mark.parametrize("solver", [StormerVerlet(), LeapfrogSolver()])
    def test_symplectic_solvers_reject_nonfinite_rhs(
        self, solver: StormerVerlet | LeapfrogSolver
    ) -> None:
        """A non-finite right-hand side is rejected."""
        with pytest.raises(ValueError, match="finite"):
            solver.step(
                lambda _t, _y: np.array([0.0, np.nan]),
                np.array([1.0, 0.0]),
                0.0,
                0.01,
            )

    @pytest.mark.parametrize("solver", [StormerVerlet(), LeapfrogSolver()])
    def test_symplectic_solvers_reject_nonfinite_candidate_state(
        self, solver: StormerVerlet | LeapfrogSolver
    ) -> None:
        """A non-finite next state is rejected."""

        def rhs(_t: float, _y: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.array([1e308, 1e308])

        with pytest.raises(ValueError, match="non-finite state"):
            solver.step(rhs, np.array([1.0, 0.0]), 0.0, 2.0)
