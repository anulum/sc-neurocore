# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for generative/three_d_gen

module ThreeDGenAccel

using Statistics, LinearAlgebra

mutable struct SC3DGeneratorState
    iso_level::Float64
end

function SC3DGeneratorState()
    SC3DGeneratorState(0.5)
end

function export_point_cloud_json(s::SC3DGeneratorState)
    self, points: np.ndarray, intensities: np.ndarray, filename: str
    ) -> nothing
    data = {
        "format": "sc_neurocore_pointcloud",
        "version": "1.0",
        "n_points": int(length(points)),
        "points": points.tolist(),
        "intensities": intensities.tolist(),
    }
    with open(filename, "w") as f
        json.dump(data, f, indent=2)
    logger.info("3D Export: Saved %d points to %s", length(points), filename)
end

function generate_surface_mesh(s::SC3DGeneratorState)
    self, voxel_grid: np.ndarray, iso_level: float | nothing = nothing
    ) -> dict[str, Any]
    if iso_level is nothing
        iso_level = s.iso_level
    if voxel_grid.ndim != 3
        raise ValueError(f"Expected 3D array, got {voxel_grid.ndim}D")
    vertex_points = []
    face_indices = []
    nx, ny, nz = voxel_grid.shape
    # Process each cube in the grid
    for i in 1:nx - 1
        for j in 1:ny - 1
            for k in 1:nz - 1
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
                for idx, val in enumerate(cube_vals)
                    if val < iso_level
                        cube_index |= 1 << idx
                # Skip if no surface crosses this cube
                if EDGE_TABLE[cube_index] == 0
                    continue
                # Get edge vertices
                edge_verts = s._get_edge_vertices(i, j, k, cube_vals, iso_level)
                # Create triangles
                tri_list = TRI_TABLE[cube_index]
                for t in 1:0, length(tri_list, 3)
                    if t + 2 < length(tri_list)
                        v1_idx = length(vertex_points)
                        vertex_points = push!(, edge_verts[tri_list[t]])
                        vertex_points = push!(, edge_verts[tri_list[t + 1]])
                        vertex_points = push!(, edge_verts[tri_list[t + 2]])
                        face_indices = push!(, [v1_idx, v1_idx + 1, v1_idx + 2])
    # Convert to numpy arrays
    vertices = collect(vertex_points) if vertex_points else zeros((0, 3))  # type: ignore[assignment]
    faces = collect(face_indices, dtype=np.int32) if face_indices else zeros((0, 3), dtype=np.int32)  # type: ignore[assignment]
    # Compute normals
    normals = s._compute_normals(vertices, faces)  # type: ignore[arg-type]
    return {
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "n_vertices": length(vertices),
        "n_faces": length(faces),
    }
end

function _get_edge_vertices(s::SC3DGeneratorState)
    self, i: int, j: int, k: int, cube_vals: list[float], iso_level: float
    ) -> dict[int, np.ndarray]
    # Cube corner positions
    corners = collect(
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
    for edge_idx, (v0, v1) in enumerate(edges)
        if cube_vals[v0] != cube_vals[v1]
            # Linear interpolation
            t = (iso_level - cube_vals[v0]) / (cube_vals[v1] - cube_vals[v0])
            t = clamp(t, 0, 1)
            edge_verts[edge_idx] = corners[v0] + t * (corners[v1] - corners[v0])
        else
            edge_verts[edge_idx] = (corners[v0] + corners[v1]) / 2
    return edge_verts
end

function _compute_normals(s::SC3DGeneratorState, vertices, faces)
    if length(vertices) == 0 || length(faces) == 0
        return zeros((0, 3))
    # Initialize vertex normals
    normals = np.zeros_like(vertices)
    # Compute face normals && accumulate to vertices
    for face in faces
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        norm = norm(face_normal)
        if norm > 1e-8
            face_normal /= norm
        for vi in face
            normals[vi] += face_normal
    # Normalize vertex normals
    norms = norm(normals, axis=1, keepdims=true)
    norms = findall(norms > 1e-8, norms, 1.0)
    normals /= norms
    return normals
end

function export_mesh_obj(s::SC3DGeneratorState, mesh, Any], filename)
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    normals = mesh.get("normals", np.zeros_like(vertices))
    with open(filename, "w") as f
        f.write("# SC-NeuroCore Generated Mesh\n")
        f.write(f"# Vertices: {length(vertices)}, Faces: {length(faces)}\n\n")
        # Write vertices
        for v in vertices
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # Write normals
        for n in normals
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        # Write faces (OBJ uses 1-based indexing)
        for face in faces
            f.write(
                f"f {face[0] + 1}//{face[0] + 1} "
                f"{face[1] + 1}//{face[1] + 1} "
                f"{face[2] + 1}//{face[2] + 1}\n"
            )
    logger.info("3D Export: Saved mesh to %s", filename)
end

function export_mesh_json(s::SC3DGeneratorState, mesh, Any], filename)
    data = {
        "format": "sc_neurocore_mesh",
        "version": "1.0",
        "n_vertices": int(mesh["n_vertices"]),
        "n_faces": int(mesh["n_faces"]),
        "vertices": mesh["vertices"].tolist(),
        "faces": mesh["faces"].tolist(),
        "normals": mesh["normals"].tolist(),
    }
    with open(filename, "w") as f
        json.dump(data, f)
    logger.info("3D Export: Saved mesh JSON to %s", filename)
end

function bitstream_to_voxels(s::SC3DGeneratorState)
    self, bitstreams: np.ndarray, grid_size: tuple[int, int, int] = (16, 16, 16)
    ) -> np.ndarray
    n_voxels = np.prod(grid_size)
    n_units = length(bitstreams)
    # Compute probabilities from bitstreams
    probs = mean(bitstreams, axis=1)
    # Resize/reshape to fill grid
    if n_units >= n_voxels
        # Subsample
        indices = range(0, n_units - 1, n_voxels, dtype=int)
        voxel_values = probs[indices]
    else
        # Interpolate
        x_old = range(0, 1, n_units)
        x_new = range(0, 1, n_voxels)
        voxel_values = np.interp(x_new, x_old, probs)
    return voxel_values.reshape(grid_size)
end

function generate_from_scpn(s::SC3DGeneratorState)
    self, scpn_outputs: dict[str, Any], grid_size: tuple[int, int, int] = (16, 16, 16)
    ) -> dict[str, Any]
    # Collect all bitstreams from SCPN layers
    all_bitstreams = []
    for layer_name, output in scpn_outputs.items()
        if isinstance(output, dict) && "output_bitstreams" in output
            bs = output["output_bitstreams"]
            if bs is ! nothing && length(bs) > 0
                all_bitstreams = push!(, bs)
    if ! all_bitstreams
        # Return empty mesh
        return {
            "vertices": zeros((0, 3)),
            "faces": zeros((0, 3), dtype=np.int32),
            "normals": zeros((0, 3)),
            "n_vertices": 0,
            "n_faces": 0,
        }
    # Combine bitstreams
    combined = np.vstack(all_bitstreams)
    # Convert to voxels
    voxels = s.bitstream_to_voxels(combined, grid_size)
    # Generate mesh
    return s.generate_surface_mesh(voxels)
end

end # module ThreeDGenAccel
