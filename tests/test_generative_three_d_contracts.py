# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for 3D generator contracts

"""Contracts for 3D generator export and SCPN-derived mesh boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.generative.three_d_gen import SC3DGenerator


def _sphere_grid() -> np.ndarray:
    grid = np.zeros((8, 8, 8))
    for i in range(8):
        for j in range(8):
            for k in range(8):
                grid[i, j, k] = (
                    1.0 if (i - 3.5) ** 2 + (j - 3.5) ** 2 + (k - 3.5) ** 2 < 6.0 else 0.0
                )
    return grid


def test_point_cloud_json_export_writes_file(tmp_path) -> None:
    generator = SC3DGenerator()
    out = tmp_path / "point_cloud.json"

    generator.export_point_cloud_json(
        np.array([[0, 0, 0], [1, 1, 1]], dtype=float),
        np.array([0.5, 0.9]),
        str(out),
    )

    assert out.stat().st_size > 0


def test_mesh_exports_write_obj_and_json_files(tmp_path) -> None:
    generator = SC3DGenerator(iso_level=0.3)
    mesh = generator.generate_surface_mesh(_sphere_grid())
    obj = tmp_path / "mesh.obj"
    json_path = tmp_path / "mesh.json"

    generator.export_mesh_obj(mesh, str(obj))
    generator.export_mesh_json(mesh, str(json_path))

    assert obj.stat().st_size > 0
    assert json_path.stat().st_size > 0


def test_scpn_generation_handles_empty_and_bitstream_inputs() -> None:
    generator = SC3DGenerator(iso_level=0.4)

    empty = generator.generate_from_scpn({})
    generated = generator.generate_from_scpn(
        {"l1": {"output_bitstreams": np.random.randint(0, 2, (64, 32)).astype(np.uint8)}},
        grid_size=(8, 8, 8),
    )

    assert empty["n_vertices"] == 0
    assert isinstance(generated, dict)


def test_surface_mesh_rejects_non_3d_grid() -> None:
    generator = SC3DGenerator()

    with pytest.raises(ValueError, match="Expected 3D"):
        generator.generate_surface_mesh(np.zeros((4, 4)))


def test_compute_normals_returns_empty_for_degenerate_mesh() -> None:
    generator = SC3DGenerator()

    normals = generator._compute_normals(np.zeros((0, 3)), np.zeros((0, 3), dtype=int))

    assert normals.shape == (0, 3)


def test_bitstream_to_voxels_subsamples_when_units_exceed_voxels() -> None:
    # 100 bitstreams mapped onto an 8-voxel grid takes the subsampling branch.
    generator = SC3DGenerator()
    bitstreams = np.random.default_rng(0).integers(0, 2, size=(100, 8)).astype(np.uint8)

    voxels = generator.bitstream_to_voxels(bitstreams, grid_size=(2, 2, 2))

    assert voxels.shape == (2, 2, 2)
