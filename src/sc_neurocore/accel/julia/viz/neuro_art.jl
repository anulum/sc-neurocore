# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for viz/neuro_art

module NeuroArtAccel

using Statistics, LinearAlgebra

mutable struct NeuroArtGeneratorState
    resolution::Float64
end

function NeuroArtGeneratorState()
    NeuroArtGeneratorState(256.0)
end

function generate_visual(s::NeuroArtGeneratorState, state_vector, Any])
    # Seed random generator with state hash to be deterministic per state
    # but chaotic
    seed = int(sum(abs(state_vector)) * 10000) % (2^32)
    rng = np.random.default_rng(seed)
    # Create base canvas
    img = zeros((s.resolution, s.resolution, 3), dtype=np.uint8)
    # 'Painters' driven by state elements
    num_painters = min(10, length(state_vector))
    for i in 1:num_painters
        val = state_vector[i]
        # Map value to color
        color = rng.integers(0, 255, 3)
        # Map value to position/size
        x = rng.integers(0, s.resolution)
        y = rng.integers(0, s.resolution)
        radius = int(abs(val) * 50) + 5
        # Draw circle (naive)
        y_grid, x_grid = np.ogrid[: s.resolution, : s.resolution]
        mask = (x_grid - x) ^ 2 + (y_grid - y) ^ 2 <= radius^2
        img[mask] = color
    return img
end

end # module NeuroArtAccel
