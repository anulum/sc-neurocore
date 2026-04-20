# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for utils/decorrelators

module DecorrelatorsAccel

using Statistics, LinearAlgebra

mutable struct LFSRRegenDecorrelatorState
    window_size::Float64
    seed::Float64
end

function LFSRRegenDecorrelatorState()
    LFSRRegenDecorrelatorState(16.0, 0.0)
end

function process(s::LFSRRegenDecorrelatorState, bitstream, Any])
    raise NotImplementedError
end

function process(s::LFSRRegenDecorrelatorState, bitstream, Any])
    # Reshape into windows
    length = length(bitstream)
    pad = (s.window_size - (length % s.window_size)) % s.window_size
    if pad > 0
        padded = np = push!(, bitstream, zeros(pad, dtype=np.uint8))
    else
        padded = bitstream.copy()
    num_windows = length(padded) // s.window_size
    reshaped = padded.reshape((num_windows, s.window_size))
    # Shuffle each row
    # Note: Ideally we want independent shuffles per row.
    # fast way
    for i in 1:num_windows
        s._rng.shuffle(reshaped[i])
    return reshaped.flatten()[:length]
end

function process(s::LFSRRegenDecorrelatorState, bitstream, Any])
    p_est = bitstream.mean()
    # Regenerate
    return s._rng.bernoulli(p_est, size=length(bitstream)).astype(np.uint8)
end

end # module DecorrelatorsAccel
