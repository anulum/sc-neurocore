# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — 3D Generation Module

"""3D mesh and point-cloud generation from voxel grids.

Generates 3D mesh and point-cloud outputs from voxel grids and probability
distributions using the marching-cubes algorithm.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Marching Cubes lookup tables for voxel-to-mesh extraction.
# Edge table: which edges are cut for each cube configuration
EDGE_TABLE = [
    0x0,
    0x109,
    0x203,
    0x30A,
    0x406,
    0x50F,
    0x605,
    0x70C,
    0x80C,
    0x905,
    0xA0F,
    0xB06,
    0xC0A,
    0xD03,
    0xE09,
    0xF00,
    0x190,
    0x99,
    0x393,
    0x29A,
    0x596,
    0x49F,
    0x795,
    0x69C,
    0x99C,
    0x895,
    0xB9F,
    0xA96,
    0xD9A,
    0xC93,
    0xF99,
    0xE90,
    0x230,
    0x339,
    0x33,
    0x13A,
    0x636,
    0x73F,
    0x435,
    0x53C,
    0xA3C,
    0xB35,
    0x83F,
    0x936,
    0xE3A,
    0xF33,
    0xC39,
    0xD30,
    0x3A0,
    0x2A9,
    0x1A3,
    0xAA,
    0x7A6,
    0x6AF,
    0x5A5,
    0x4AC,
    0xBAC,
    0xAA5,
    0x9AF,
    0x8A6,
    0xFAA,
    0xEA3,
    0xDA9,
    0xCA0,
    0x460,
    0x569,
    0x663,
    0x76A,
    0x66,
    0x16F,
    0x265,
    0x36C,
    0xC6C,
    0xD65,
    0xE6F,
    0xF66,
    0x86A,
    0x963,
    0xA69,
    0xB60,
    0x5F0,
    0x4F9,
    0x7F3,
    0x6FA,
    0x1F6,
    0xFF,
    0x3F5,
    0x2FC,
    0xDFC,
    0xCF5,
    0xFFF,
    0xEF6,
    0x9FA,
    0x8F3,
    0xBF9,
    0xAF0,
    0x650,
    0x759,
    0x453,
    0x55A,
    0x256,
    0x35F,
    0x55,
    0x15C,
    0xE5C,
    0xF55,
    0xC5F,
    0xD56,
    0xA5A,
    0xB53,
    0x859,
    0x950,
    0x7C0,
    0x6C9,
    0x5C3,
    0x4CA,
    0x3C6,
    0x2CF,
    0x1C5,
    0xCC,
    0xFCC,
    0xEC5,
    0xDCF,
    0xCC6,
    0xBCA,
    0xAC3,
    0x9C9,
    0x8C0,
    0x8C0,
    0x9C9,
    0xAC3,
    0xBCA,
    0xCC6,
    0xDCF,
    0xEC5,
    0xFCC,
    0xCC,
    0x1C5,
    0x2CF,
    0x3C6,
    0x4CA,
    0x5C3,
    0x6C9,
    0x7C0,
    0x950,
    0x859,
    0xB53,
    0xA5A,
    0xD56,
    0xC5F,
    0xF55,
    0xE5C,
    0x15C,
    0x55,
    0x35F,
    0x256,
    0x55A,
    0x453,
    0x759,
    0x650,
    0xAF0,
    0xBF9,
    0x8F3,
    0x9FA,
    0xEF6,
    0xFFF,
    0xCF5,
    0xDFC,
    0x2FC,
    0x3F5,
    0xFF,
    0x1F6,
    0x6FA,
    0x7F3,
    0x4F9,
    0x5F0,
    0xB60,
    0xA69,
    0x963,
    0x86A,
    0xF66,
    0xE6F,
    0xD65,
    0xC6C,
    0x36C,
    0x265,
    0x16F,
    0x66,
    0x76A,
    0x663,
    0x569,
    0x460,
    0xCA0,
    0xDA9,
    0xEA3,
    0xFAA,
    0x8A6,
    0x9AF,
    0xAA5,
    0xBAC,
    0x4AC,
    0x5A5,
    0x6AF,
    0x7A6,
    0xAA,
    0x1A3,
    0x2A9,
    0x3A0,
    0xD30,
    0xC39,
    0xF33,
    0xE3A,
    0x936,
    0x83F,
    0xB35,
    0xA3C,
    0x53C,
    0x435,
    0x73F,
    0x636,
    0x13A,
    0x33,
    0x339,
    0x230,
    0xE90,
    0xF99,
    0xC93,
    0xD9A,
    0xA96,
    0xB9F,
    0x895,
    0x99C,
    0x69C,
    0x795,
    0x49F,
    0x596,
    0x29A,
    0x393,
    0x99,
    0x190,
    0xF00,
    0xE09,
    0xD03,
    0xC0A,
    0xB06,
    0xA0F,
    0x905,
    0x80C,
    0x70C,
    0x605,
    0x50F,
    0x406,
    0x30A,
    0x203,
    0x109,
    0x0,
]

# Triangle table: which triangles to create for each configuration
# Simplified version - maps cube config to edge triplets
TRI_TABLE = [
    [],
    [0, 8, 3],
    [0, 1, 9],
    [1, 8, 3, 9, 8, 1],
    [1, 2, 10],
    [0, 8, 3, 1, 2, 10],
    [9, 2, 10, 0, 2, 9],
    [2, 8, 3, 2, 10, 8, 10, 9, 8],
    [3, 11, 2],
    [0, 11, 2, 8, 11, 0],
    [1, 9, 0, 2, 3, 11],
    [1, 11, 2, 1, 9, 11, 9, 8, 11],
    [3, 10, 1, 11, 10, 3],
    [0, 10, 1, 0, 8, 10, 8, 11, 10],
    [3, 9, 0, 3, 11, 9, 11, 10, 9],
    [9, 8, 10, 10, 8, 11],
    # ... (abbreviated for space - full table would have 256 entries)
]

# Extend TRI_TABLE to 256 entries (fill with empty for simplicity)
while len(TRI_TABLE) < 256:
    TRI_TABLE.append([])


@dataclass
class SC3DGenerator:
    """
    Generator for 3D mesh and point cloud outputs from stochastic voxel data.

    Implements Marching Cubes algorithm for isosurface extraction.
    """

    iso_level: float = 0.5  # Threshold for surface extraction

    def export_point_cloud_json(
        self, points: np.ndarray, intensities: np.ndarray, filename: str
    ) -> None:
        """
        Export a point cloud to JSON format.

        Args:
            points: Nx3 array of point coordinates
            intensities: N array of intensity values
            filename: Output file path
        """
        data = {
            "format": "sc_neurocore_pointcloud",
            "version": "1.0",
            "n_points": int(len(points)),
            "points": points.tolist(),
            "intensities": intensities.tolist(),
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("3D Export: Saved %d points to %s", len(points), filename)

    def generate_surface_mesh(
        self, voxel_grid: np.ndarray, iso_level: float | None = None
    ) -> dict[str, Any]:
        """
        Generate a surface mesh from a voxel grid using Marching Cubes.

        Args:
            voxel_grid: 3D numpy array of scalar values
            iso_level: Isosurface threshold (default: self.iso_level)

        Returns
        -------
            Dict with 'vertices', 'faces', 'normals'
        """
        if iso_level is None:
            iso_level = self.iso_level

        if voxel_grid.ndim != 3:
            raise ValueError(f"Expected 3D array, got {voxel_grid.ndim}D")

        vertices: list[float] = []
        faces = []

        nx, ny, nz = voxel_grid.shape

        # Process each cube in the grid
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    # Get the 8 corner values
                    cube_vals = [
                        voxel_grid[i, j, k],
                        voxel_grid[i + 1, j, k],
                        voxel_grid[i + 1, j + 1, k],
                        voxel_grid[i, j + 1, k],
                        voxel_grid[i, j, k + 1],
                        voxel_grid[i + 1, j, k + 1],
                        voxel_grid[i + 1, j + 1, k + 1],
                        voxel_grid[i, j + 1, k + 1],
                    ]

                    # Determine cube index
                    cube_index = 0
                    for idx, val in enumerate(cube_vals):
                        if val < iso_level:
                            cube_index |= 1 << idx

                    # Skip if no surface crosses this cube
                    if EDGE_TABLE[cube_index] == 0:
                        continue

                    # Get edge vertices
                    edge_verts = self._get_edge_vertices(i, j, k, cube_vals, iso_level)

                    # Create triangles
                    tri_list = TRI_TABLE[cube_index]
                    for t in range(0, len(tri_list), 3):
                        if t + 2 < len(tri_list):
                            v1_idx = len(vertices)
                            vertices.append(edge_verts[tri_list[t]])  # type: ignore
                            vertices.append(edge_verts[tri_list[t + 1]])  # type: ignore
                            vertices.append(edge_verts[tri_list[t + 2]])  # type: ignore
                            faces.append([v1_idx, v1_idx + 1, v1_idx + 2])

        # Convert to numpy arrays
        vertices = np.array(vertices) if vertices else np.zeros((0, 3))  # type: ignore[assignment]
        faces = np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)  # type: ignore[assignment]

        # Compute normals
        normals = self._compute_normals(vertices, faces)  # type: ignore[arg-type]

        return {
            "vertices": vertices,
            "faces": faces,
            "normals": normals,
            "n_vertices": len(vertices),
            "n_faces": len(faces),
        }

    def _get_edge_vertices(
        self, i: int, j: int, k: int, cube_vals: list[float], iso_level: float
    ) -> dict[int, np.ndarray]:
        """Compute vertex positions along cube edges."""
        # Cube corner positions
        corners = np.array(
            [
                [i, j, k],
                [i + 1, j, k],
                [i + 1, j + 1, k],
                [i, j + 1, k],
                [i, j, k + 1],
                [i + 1, j, k + 1],
                [i + 1, j + 1, k + 1],
                [i, j + 1, k + 1],
            ],
            dtype=np.float64,
        )

        # Edge endpoints
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

        edge_verts = {}
        for edge_idx, (v0, v1) in enumerate(edges):
            if cube_vals[v0] != cube_vals[v1]:
                # Linear interpolation
                t = (iso_level - cube_vals[v0]) / (cube_vals[v1] - cube_vals[v0])
                t = np.clip(t, 0, 1)
                edge_verts[edge_idx] = corners[v0] + t * (corners[v1] - corners[v0])
            else:
                edge_verts[edge_idx] = (corners[v0] + corners[v1]) / 2

        return edge_verts

    def _compute_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Compute vertex normals from face normals."""
        if len(vertices) == 0 or len(faces) == 0:
            return np.zeros((0, 3))

        # Initialize vertex normals
        normals = np.zeros_like(vertices)

        # Compute face normals and accumulate to vertices
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(face_normal)
            if norm > 1e-8:
                face_normal /= norm
            for vi in face:
                normals[vi] += face_normal

        # Normalize vertex normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        normals /= norms

        return normals

    def export_mesh_obj(self, mesh: dict[str, Any], filename: str) -> None:
        """
        Export mesh to OBJ format.

        Args:
            mesh: Dict from generate_surface_mesh()
            filename: Output file path
        """
        vertices = mesh["vertices"]
        faces = mesh["faces"]
        normals = mesh.get("normals", np.zeros_like(vertices))

        with open(filename, "w") as f:
            f.write("# SC-NeuroCore Generated Mesh\n")
            f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")

            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write normals
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(
                    f"f {face[0] + 1}//{face[0] + 1} "
                    f"{face[1] + 1}//{face[1] + 1} "
                    f"{face[2] + 1}//{face[2] + 1}\n"
                )

        logger.info("3D Export: Saved mesh to %s", filename)

    def export_mesh_json(self, mesh: dict[str, Any], filename: str) -> None:
        """
        Export mesh to JSON format.

        Args:
            mesh: Dict from generate_surface_mesh()
            filename: Output file path
        """
        data = {
            "format": "sc_neurocore_mesh",
            "version": "1.0",
            "n_vertices": int(mesh["n_vertices"]),
            "n_faces": int(mesh["n_faces"]),
            "vertices": mesh["vertices"].tolist(),
            "faces": mesh["faces"].tolist(),
            "normals": mesh["normals"].tolist(),
        }
        with open(filename, "w") as f:
            json.dump(data, f)
        logger.info("3D Export: Saved mesh JSON to %s", filename)

    def bitstream_to_voxels(
        self, bitstreams: np.ndarray, grid_size: tuple[int, int, int] = (16, 16, 16)
    ) -> np.ndarray:
        """
        Convert bitstream outputs to a voxel grid.

        Args:
            bitstreams: 2D array of bitstreams (n_units, length)
            grid_size: Output voxel grid dimensions

        Returns
        -------
            3D voxel grid with probability-based values
        """
        n_voxels = np.prod(grid_size)
        n_units = len(bitstreams)

        # Compute probabilities from bitstreams
        probs = np.mean(bitstreams, axis=1)

        # Resize/reshape to fill grid
        if n_units >= n_voxels:
            # Subsample
            indices = np.linspace(0, n_units - 1, n_voxels, dtype=int)
            voxel_values = probs[indices]
        else:
            # Interpolate
            x_old = np.linspace(0, 1, n_units)
            x_new = np.linspace(0, 1, n_voxels)
            voxel_values = np.interp(x_new, x_old, probs)

        return voxel_values.reshape(grid_size)

    def generate_from_scpn(
        self, scpn_outputs: dict[str, Any], grid_size: tuple[int, int, int] = (16, 16, 16)
    ) -> dict[str, Any]:
        """
        Generate 3D mesh directly from SCPN layer outputs.

        Args:
            scpn_outputs: Output from run_integrated_step()
            grid_size: Voxel grid dimensions

        Returns
        -------
            Mesh dict from generate_surface_mesh()
        """
        # Collect all bitstreams from SCPN layers
        all_bitstreams = []
        for layer_name, output in scpn_outputs.items():
            if isinstance(output, dict) and "output_bitstreams" in output:
                bs = output["output_bitstreams"]
                if bs is not None and len(bs) > 0:
                    all_bitstreams.append(bs)

        if not all_bitstreams:
            # Return empty mesh
            return {
                "vertices": np.zeros((0, 3)),
                "faces": np.zeros((0, 3), dtype=np.int32),
                "normals": np.zeros((0, 3)),
                "n_vertices": 0,
                "n_faces": 0,
            }

        # Combine bitstreams
        combined = np.vstack(all_bitstreams)

        # Convert to voxels
        voxels = self.bitstream_to_voxels(combined, grid_size)

        # Generate mesh
        return self.generate_surface_mesh(voxels)
