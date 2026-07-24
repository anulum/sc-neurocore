# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMeepAdapterContract from former test_fdtd.py

"""Focused suite: TestMeepAdapterContract from former test_fdtd.py."""

from __future__ import annotations

from fdtd_support import *  # noqa: F403


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
