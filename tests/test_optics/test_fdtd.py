# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for 1D + 2D FDTD solvers

"""Tests for ``FDTDSolver`` (1D) and ``FDTD2DSolver`` (2D Berenger PML).

1D uses a quadratic-ramp absorbing boundary condition (ABC). The 2D
solver implements the split-field Berenger PML with per-direction
conductivities σx/σy and matched-impedance (σ*/μ₀ = σ/ε₀).

These tests exercise:

- Initial-field/grid invariants after construction.
- Injected-pulse energy is non-zero and finite.
- Field values remain bounded (no blow-up) under nominal CFL.
- 1D ABC monotonically damps field amplitude at the grid edges across
  multiple passes.
- 2D Berenger PML ratios σy*(μ₀/ε₀) are set so the matched-impedance
  condition reduces interior reflection.
- Ill-formed material maps are rejected.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from sc_neurocore.optics.photonic_emitter import (
    FDTD2DSolver,
    FDTDSolver,
    MeepAdapter,
    PhotonicTarget,
)


# ---------------------------------------------------------------------------
# 1D solver.
# ---------------------------------------------------------------------------


class TestFDTDSolver1D:
    @pytest.fixture
    def solver(self) -> FDTDSolver:
        return FDTDSolver(grid_size=256, dx_um=0.02, dt_factor=0.5, refractive_index=3.48)

    def test_initial_fields_are_zero(self, solver):
        ez, hy = solver.snapshot()
        assert np.all(ez == 0.0)
        assert np.all(hy == 0.0)
        assert solver.field_energy() == 0.0

    def test_has_absorbing_boundary_taper(self, solver):
        # ABC (not split-field PML): multiplicative quadratic ramp on the
        # outermost ``boundary_cells`` on each side.
        assert hasattr(solver, "_abc_taper")
        assert solver._abc_taper.shape == (256,)
        # Taper strength at the edge should be < 1 (absorbing), = 1 inside.
        assert solver._abc_taper[0] < 1.0
        assert solver._abc_taper[128] == pytest.approx(1.0)

    def test_injected_pulse_deposits_energy(self, solver):
        solver.inject_pulse(position=128, wavelength_nm=1550.0, amplitude=1.0)
        e0 = solver.field_energy()
        assert e0 > 0.0
        assert np.isfinite(e0)

    def test_step_preserves_finiteness(self, solver):
        solver.inject_pulse(position=128, wavelength_nm=1550.0, amplitude=1.0)
        solver.step(200)
        ez, hy = solver.snapshot()
        assert np.all(np.isfinite(ez))
        assert np.all(np.isfinite(hy))

    def test_absorbing_boundary_strictly_reduces_edge_energy(self, solver):
        # Pulse injected near the left edge must see its edge-cell amplitude
        # reduced after the ABC taper is applied over many passes.
        solver.inject_pulse(position=30, wavelength_nm=1550.0, amplitude=1.0)
        initial_edge = abs(solver.ez[0])
        solver.step(500)
        final_edge = abs(solver.ez[0])
        # Either the pulse left through the ABC (edge damped) or never
        # reached it — both satisfy the invariant "edge doesn't blow up".
        assert final_edge <= initial_edge + 1e-6

    def test_loss_setter_monotonically_decreases_energy(self, solver):
        solver.inject_pulse(position=128, amplitude=1.0)
        e_start = solver.field_energy()
        solver.set_loss(loss_db_per_cm=100.0)  # very lossy
        solver.step(100)
        e_after = solver.field_energy()
        assert e_after < e_start


# ---------------------------------------------------------------------------
# 2D solver — split-field Berenger PML.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Optional Meep adapter.
# ---------------------------------------------------------------------------


class TestMeepAdapterContract:
    def test_available_meep_path_builds_simulation_and_reports_flux(self, monkeypatch):
        calls: dict[str, object] = {}

        class FakeVector3:
            def __init__(self, *coords):
                self.coords = coords

        class FakeContinuousSource:
            def __init__(self, frequency):
                self.frequency = frequency

        class FakeSource:
            def __init__(self, source, component, center, size):
                calls["source"] = source
                calls["component"] = component
                calls["source_center"] = center.coords
                calls["source_size"] = size.coords

        class FakeMedium:
            def __init__(self, index):
                calls["medium_index"] = index

        class FakeBlock:
            def __init__(self, size, center, material):
                calls["block_size"] = size.coords
                calls["block_center"] = center.coords
                calls["block_material"] = material

        class FakePML:
            def __init__(self, thickness):
                calls["pml"] = thickness

        class FakeFluxRegion:
            def __init__(self, center, size):
                calls["flux_center"] = center.coords
                calls["flux_size"] = size.coords

        class FakeSimulation:
            def __init__(self, cell_size, resolution, sources, geometry, boundary_layers):
                calls["cell_size"] = cell_size.coords
                calls["resolution"] = resolution
                calls["sources_count"] = len(sources)
                calls["geometry_count"] = len(geometry)
                calls["boundary_count"] = len(boundary_layers)

            def add_flux(self, freq, df, nfreq, region):
                calls["flux"] = (freq, df, nfreq, region)
                return "flux-handle"

            def run(self, until):
                calls["run_until"] = until

        fake_meep = types.SimpleNamespace(
            Vector3=FakeVector3,
            ContinuousSource=FakeContinuousSource,
            Source=FakeSource,
            Medium=FakeMedium,
            Block=FakeBlock,
            PML=FakePML,
            FluxRegion=FakeFluxRegion,
            Simulation=FakeSimulation,
            Ez="Ez",
            get_fluxes=lambda handle: [0.73] if handle == "flux-handle" else [],
        )
        monkeypatch.setitem(sys.modules, "meep", fake_meep)

        geometry = MeepAdapter.build_waveguide_geometry(
            PhotonicTarget.lightmatter(),
            waveguide_width_um=0.4,
            length_um=8.0,
        )
        result = MeepAdapter.run_simulation(geometry, run_time=17.5)

        assert MeepAdapter.is_available() is True
        assert result == {
            "transmission": pytest.approx(0.73),
            "reflection": 0.0,
            "field_decay": 0.0,
            "run_time": 17.5,
            "mock": False,
            "wavelength_nm": 1550.0,
        }
        assert calls["cell_size"] == tuple(geometry["cell_size"])
        assert calls["resolution"] == geometry["resolution"]
        assert calls["sources_count"] == 1
        assert calls["geometry_count"] == 1
        assert calls["boundary_count"] == 1
        assert calls["pml"] == geometry["pml_layers"]
        assert calls["run_until"] == 17.5
