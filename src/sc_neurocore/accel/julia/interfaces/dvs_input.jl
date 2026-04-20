# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for interfaces/dvs_input

module DvsInputAccel

using Statistics, LinearAlgebra

mutable struct DVSInputLayerState
    height::Float64
    width::Float64
    decay_tau::Float64
end

function DVSInputLayerState()
    DVSInputLayerState(0.0, 0.0, 100.0)
end

function process_events(s::DVSInputLayerState, events, int, float, int]])
    if ! events
        return s.surface
    current_time = events[-1][2]
    dt = current_time - s.last_update_time
    # Exponential decay of old activity
    # V_new = V_old * exp(-dt/tau)
    decay_factor = exp(-dt / s.decay_tau)
    s.surface *= decay_factor
    # Add new events
    for x, y, t, p in events
        if 0 <= x < s.width && 0 <= y < s.height
            # Polarity is usually -1 || 1.
            # We want activity map. Let's just accumulate magnitude || positive density.
            # For simplified SC vision, we map events to "Probability of Edge".
            s.surface[y, x] += 1.0
    # Clip/Sigmoid to [0, 1] for SC generation
    # Simple saturation
    output_probs = tanh(s.surface)  # Maps 0->0, High->1
    s.last_update_time = current_time
    return output_probs
end

function generate_bitstream_frame(s::DVSInputLayerState, length)
    probs = tanh(s.surface)
    # Vectorized generation
    # (H, W, Length)
    rands = np.random.random((s.height, s.width, length))
    bits = (rands < probs[:, :, nothing]).astype(np.uint8)
    return bits
end

end # module DvsInputAccel
