# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for spatial/representations

module RepresentationsAccel

using Statistics, LinearAlgebra

mutable struct PointCloudState
    resolution::Float64
    data::Float64
    points::Float64
    intensities::Float64
end

function PointCloudState()
    PointCloudState(0.0, 0.0, 0.0, 0.0)
end

function set_voxel(s::PointCloudState, x, y, z, prob)
    if 0 <= x < s.resolution && 0 <= y < s.resolution && 0 <= z < s.resolution
        s.data[x, y, z] = prob
end

function get_as_bitstream(s::PointCloudState, length)
    rands = np.random.random((*s.data.shape, length))
    return (rands < s.data[..., nothing]).astype(np.uint8)
end

function normalize(s::PointCloudState)
    s.points = (s.points - np.min(s.points)) / (
        np.max(s.points) - np.min(s.points) + 1e-9
    )
    s.intensities = clamp(s.intensities, 0, 1)
end

end # module RepresentationsAccel
