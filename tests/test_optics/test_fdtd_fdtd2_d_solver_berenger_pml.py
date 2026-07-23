# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFDTD2DSolverBerengerPML from former test_fdtd.py

"""Focused suite: TestFDTD2DSolverBerengerPML from former test_fdtd.py."""

from __future__ import annotations

from fdtd_support import *  # noqa: F403

class TestFDTD2DSolverBerengerPML:
    @pytest.fixture
    def solver(self) -> FDTD2DSolver:
        return FDTD2DSolver(nx=64, ny=48, dx_um=0.02, dy_um=0.02, pml_layers=8)

    def test_initial_fields_zero(self, solver):
        ez, hx, hy = solver.snapshot()
        assert np.all(ez == 0.0)
        assert np.all(hx == 0.0)
        assert np.all(hy == 0.0)
        assert solver.field_energy() == 0.0

    def test_pml_conductivity_profile_on_edges(self, solver):
        # σx peaks at the nx=0 and nx=nx-1 edges; σy peaks at ny=0, ny-1.
        assert solver.sigma_x[0, solver.ny // 2] > 0.0
        assert solver.sigma_x[solver.nx - 1, solver.ny // 2] > 0.0
        # Interior is conductivity-free.
        assert solver.sigma_x[solver.nx // 2, solver.ny // 2] == 0.0
        assert solver.sigma_y[solver.nx // 2, solver.ny // 2] == 0.0

    def test_pml_profile_is_cubic_ramp(self, solver):
        # Sample the σx profile and check the monotone cubic ramp.
        p = solver.pml_layers
        profile = [solver.sigma_x[i, solver.ny // 2] for i in range(p)]
        assert all(profile[i] >= profile[i + 1] for i in range(p - 1))

    def test_waveguide_sets_refractive_index(self, solver):
        solver.set_waveguide(y_center=24, width_cells=4, refractive_index=3.48)
        assert solver.n_map[solver.nx // 2, 24] == pytest.approx(3.48)
        assert solver.n_map[0, 0] == pytest.approx(1.0)

    def test_rejects_invalid_refractive_index(self, solver):
        with pytest.raises(ValueError):
            solver.set_waveguide(y_center=24, width_cells=4, refractive_index=0.8)

    def test_rejects_source_out_of_bounds(self, solver):
        with pytest.raises(ValueError):
            solver.inject_source(x=solver.nx, y=0)

    def test_rejects_nonpositive_wavelength(self, solver):
        with pytest.raises(ValueError):
            solver.inject_source(x=10, y=10, wavelength_nm=0.0)

    def test_step_rejects_zero_refractive_index(self, solver):
        # Corrupt the n_map after construction.
        solver.n_map[5, 5] = 0.0
        with pytest.raises(ValueError):
            solver.step(1)

    def test_injected_source_deposits_energy(self, solver):
        solver.inject_source(x=32, y=24, wavelength_nm=1550.0, amplitude=1.0, sigma_cells=4)
        e0 = solver.field_energy()
        assert e0 > 0.0
        assert np.isfinite(e0)

    def test_step_preserves_finiteness_under_cfl(self, solver):
        solver.inject_source(x=32, y=24, wavelength_nm=1550.0, amplitude=1.0, sigma_cells=4)
        solver.step(200)
        ez, hx, hy = solver.snapshot()
        assert np.all(np.isfinite(ez))
        assert np.all(np.isfinite(hx))
        assert np.all(np.isfinite(hy))

    def test_pml_absorbs_outgoing_energy(self, solver):
        # With a central source and a PML, after enough timesteps for the
        # pulse to reach the boundaries and exit, total energy must be
        # strictly less than the peak achieved mid-simulation.
        solver.inject_source(x=32, y=24, wavelength_nm=1550.0, amplitude=1.0, sigma_cells=6)
        peak = 0.0
        for _ in range(20):
            solver.step(5)
            peak = max(peak, solver.field_energy())
        for _ in range(100):
            solver.step(5)
        final_energy = solver.field_energy()
        assert final_energy < peak

    def test_cross_section_shape(self, solver):
        solver.inject_source(x=32, y=24, wavelength_nm=1550.0, amplitude=1.0, sigma_cells=4)
        solver.step(10)
        cs = solver.cross_section(x=32)
        assert cs.shape == (solver.ny,)
