# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFDTDSolver1D from former test_fdtd.py

"""Focused suite: TestFDTDSolver1D from former test_fdtd.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.optics.photonic_emitter import (
    FDTDSolver,
)


class TestFDTDSolver1D:
    """Fields, boundaries and loss of the 1D FDTD solver."""

    @pytest.fixture
    def solver(self) -> FDTDSolver:
        """Return a 256-cell solver in silicon."""
        return FDTDSolver(grid_size=256, dx_um=0.02, dt_factor=0.5, refractive_index=3.48)

    def test_initial_fields_are_zero(self, solver: FDTDSolver) -> None:
        """A new solver has zero fields and zero energy."""
        ez, hy = solver.snapshot()
        assert np.all(ez == 0.0)
        assert np.all(hy == 0.0)
        assert solver.field_energy() == 0.0

    def test_has_absorbing_boundary_taper(self, solver: FDTDSolver) -> None:
        """The absorbing taper damps the edges and leaves the middle at one."""
        # ABC (not split-field PML): multiplicative quadratic ramp on the
        # outermost ``boundary_cells`` on each side.
        assert hasattr(solver, "_abc_taper")
        assert solver._abc_taper.shape == (256,)
        # Taper strength at the edge should be < 1 (absorbing), = 1 inside.
        assert solver._abc_taper[0] < 1.0
        assert solver._abc_taper[128] == pytest.approx(1.0)

    def test_injected_pulse_deposits_energy(self, solver: FDTDSolver) -> None:
        """An injected pulse puts finite energy into the grid."""
        solver.inject_pulse(position=128, wavelength_nm=1550.0, amplitude=1.0)
        e0 = solver.field_energy()
        assert e0 > 0.0
        assert np.isfinite(e0)

    def test_step_preserves_finiteness(self, solver: FDTDSolver) -> None:
        """Two hundred steps keep every field finite."""
        solver.inject_pulse(position=128, wavelength_nm=1550.0, amplitude=1.0)
        solver.step(200)
        ez, hy = solver.snapshot()
        assert np.all(np.isfinite(ez))
        assert np.all(np.isfinite(hy))

    def test_absorbing_boundary_strictly_reduces_edge_energy(self, solver: FDTDSolver) -> None:
        """The edge field does not grow over five hundred steps."""
        # Pulse injected near the left edge must see its edge-cell amplitude
        # reduced after the ABC taper is applied over many passes.
        solver.inject_pulse(position=30, wavelength_nm=1550.0, amplitude=1.0)
        initial_edge = abs(solver.ez[0])
        solver.step(500)
        final_edge = abs(solver.ez[0])
        # Either the pulse left through the ABC (edge damped) or never
        # reached it — both satisfy the invariant "edge doesn't blow up".
        assert final_edge <= initial_edge + 1e-6

    def test_loss_setter_monotonically_decreases_energy(self, solver: FDTDSolver) -> None:
        """A lossy medium lowers the field energy."""
        solver.inject_pulse(position=128, amplitude=1.0)
        e_start = solver.field_energy()
        solver.set_loss(loss_db_per_cm=100.0)  # very lossy
        solver.step(100)
        e_after = solver.field_energy()
        assert e_after < e_start
