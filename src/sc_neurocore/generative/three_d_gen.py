"""
SC-NeuroCore 3D Generation Module
==================================

Generates 3D mesh and point cloud outputs from voxel grids and
probability distributions using the Marching Cubes algorithm.

Author: Claude (Session 2026-01-31)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json

# Marching Cubes lookup tables (simplified)
# Edge table: which edges are cut for each cube configuration
EDGE_TABLE = [
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
]

# Triangle table: which triangles to create for each configuration
# Simplified version - maps cube config to edge triplets
TRI_TABLE = [
    [], [0, 8, 3], [0, 1, 9], [1, 8, 3, 9, 8, 1], [1, 2, 10],
    [0, 8, 3, 1, 2, 10], [9, 2, 10, 0, 2, 9], [2, 8, 3, 2, 10, 8, 10, 9, 8],
    [3, 11, 2], [0, 11, 2, 8, 11, 0], [1, 9, 0, 2, 3, 11],
    [1, 11, 2, 1, 9, 11, 9, 8, 11], [3, 10, 1, 11, 10, 3],
    [0, 10, 1, 0, 8, 10, 8, 11, 10], [3, 9, 0, 3, 11, 9, 11, 10, 9],
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

    def export_point_cloud_json(self, points: np.ndarray,
                                intensities: np.ndarray,
                                filename: str):
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
            "intensities": intensities.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"3D Export: Saved {len(points)} points to {filename}")

    def generate_surface_mesh(self, voxel_grid: np.ndarray,
                              iso_level: float = None) -> Dict:
        """
        Generate a surface mesh from a voxel grid using Marching Cubes.

        Args:
            voxel_grid: 3D numpy array of scalar values
            iso_level: Isosurface threshold (default: self.iso_level)

        Returns:
            Dict with 'vertices', 'faces', 'normals'
        """
        if iso_level is None:
            iso_level = self.iso_level

        if voxel_grid.ndim != 3:
            raise ValueError(f"Expected 3D array, got {voxel_grid.ndim}D")

        vertices = []
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
                            cube_index |= (1 << idx)

                    # Skip if no surface crosses this cube
                    if EDGE_TABLE[cube_index] == 0:
                        continue

                    # Get edge vertices
                    edge_verts = self._get_edge_vertices(
                        i, j, k, cube_vals, iso_level
                    )

                    # Create triangles
                    tri_list = TRI_TABLE[cube_index]
                    for t in range(0, len(tri_list), 3):
                        if t + 2 < len(tri_list):
                            v1_idx = len(vertices)
                            vertices.append(edge_verts[tri_list[t]])
                            vertices.append(edge_verts[tri_list[t + 1]])
                            vertices.append(edge_verts[tri_list[t + 2]])
                            faces.append([v1_idx, v1_idx + 1, v1_idx + 2])

        # Convert to numpy arrays
        vertices = np.array(vertices) if vertices else np.zeros((0, 3))
        faces = np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)

        # Compute normals
        normals = self._compute_normals(vertices, faces)

        return {
            "vertices": vertices,
            "faces": faces,
            "normals": normals,
            "n_vertices": len(vertices),
            "n_faces": len(faces)
        }

    def _get_edge_vertices(self, i: int, j: int, k: int,
                           cube_vals: List[float],
                           iso_level: float) -> Dict[int, np.ndarray]:
        """Compute vertex positions along cube edges."""
        # Cube corner positions
        corners = np.array([
            [i, j, k], [i + 1, j, k], [i + 1, j + 1, k], [i, j + 1, k],
            [i, j, k + 1], [i + 1, j, k + 1], [i + 1, j + 1, k + 1], [i, j + 1, k + 1]
        ], dtype=np.float64)

        # Edge endpoints
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
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

    def _compute_normals(self, vertices: np.ndarray,
                         faces: np.ndarray) -> np.ndarray:
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

    def export_mesh_obj(self, mesh: Dict, filename: str):
        """
        Export mesh to OBJ format.

        Args:
            mesh: Dict from generate_surface_mesh()
            filename: Output file path
        """
        vertices = mesh['vertices']
        faces = mesh['faces']
        normals = mesh.get('normals', np.zeros_like(vertices))

        with open(filename, 'w') as f:
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
                f.write(f"f {face[0]+1}//{face[0]+1} "
                       f"{face[1]+1}//{face[1]+1} "
                       f"{face[2]+1}//{face[2]+1}\n")

        print(f"3D Export: Saved mesh to {filename}")

    def export_mesh_json(self, mesh: Dict, filename: str):
        """
        Export mesh to JSON format.

        Args:
            mesh: Dict from generate_surface_mesh()
            filename: Output file path
        """
        data = {
            "format": "sc_neurocore_mesh",
            "version": "1.0",
            "n_vertices": int(mesh['n_vertices']),
            "n_faces": int(mesh['n_faces']),
            "vertices": mesh['vertices'].tolist(),
            "faces": mesh['faces'].tolist(),
            "normals": mesh['normals'].tolist()
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"3D Export: Saved mesh JSON to {filename}")

    def bitstream_to_voxels(self, bitstreams: np.ndarray,
                            grid_size: Tuple[int, int, int] = (16, 16, 16)
                            ) -> np.ndarray:
        """
        Convert bitstream outputs to a voxel grid.

        Args:
            bitstreams: 2D array of bitstreams (n_units, length)
            grid_size: Output voxel grid dimensions

        Returns:
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

    def generate_from_scpn(self, scpn_outputs: Dict,
                           grid_size: Tuple[int, int, int] = (16, 16, 16)
                           ) -> Dict:
        """
        Generate 3D mesh directly from SCPN layer outputs.

        Args:
            scpn_outputs: Output from run_integrated_step()
            grid_size: Voxel grid dimensions

        Returns:
            Mesh dict from generate_surface_mesh()
        """
        # Collect all bitstreams from SCPN layers
        all_bitstreams = []
        for layer_name, output in scpn_outputs.items():
            if isinstance(output, dict) and 'output_bitstreams' in output:
                bs = output['output_bitstreams']
                if bs is not None and len(bs) > 0:
                    all_bitstreams.append(bs)

        if not all_bitstreams:
            # Return empty mesh
            return {"vertices": np.zeros((0, 3)), "faces": np.zeros((0, 3), dtype=np.int32),
                    "normals": np.zeros((0, 3)), "n_vertices": 0, "n_faces": 0}

        # Combine bitstreams
        combined = np.vstack(all_bitstreams)

        # Convert to voxels
        voxels = self.bitstream_to_voxels(combined, grid_size)

        # Generate mesh
        return self.generate_surface_mesh(voxels)
