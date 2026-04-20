# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for residual/blocks

module BlocksAccel

using Statistics, LinearAlgebra

mutable struct DeepSNNStackState
    n_features::Float64
    threshold::Float64
    tau_mem::Float64
    W1::Float64
    W2::Float64
    _v::Float64
    W::Float64
    blocks::Float64
end

function DeepSNNStackState()
    DeepSNNStackState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function forward(s::DeepSNNStackState, x)
    alpha = exp(-1.0 / s.tau_mem)
    # First transform
    h = s.W1 @ x
    # LIF on hidden
    v1 = alpha * zeros(s.n_features) + (1 - alpha) * h
    s1 = (v1 >= s.threshold).astype(np.float64)
    # Second transform
    h2 = s.W2 @ s1
    # Membrane shortcut: add input directly to membrane (! spikes)
    s._v = alpha * s._v + (1 - alpha) * (h2 + x)
    spikes = (s._v >= s.threshold).astype(np.float64)
    s._v -= spikes * s.threshold
    return spikes
end

function reset(s::DeepSNNStackState)
    s._v = zeros(s.n_features)
end

function forward(s::DeepSNNStackState, x)
    h = s.W @ x
    s._v += h
    spikes = (s._v >= s.threshold).astype(np.float64)
    s._v -= spikes * s.threshold
    return clamp(spikes + x, 0, 1)
end

function reset(s::DeepSNNStackState)
    s._v = zeros(s.n_features)
end

function forward(s::DeepSNNStackState, x)
    h = x
    for block in s.blocks
        h = block.forward(h)
    return h
end

function reset(s::DeepSNNStackState)
    for block in s.blocks
        block.reset()
end

function n_blocks(s::DeepSNNStackState)
    return length(s.blocks)
end

function depth(s::DeepSNNStackState)
    return length(s.blocks) * 2
end

end # module BlocksAccel
