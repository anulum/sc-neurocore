# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for sources/bitstream_current_source

module BitstreamCurrentSourceAccel

using Statistics, LinearAlgebra

mutable struct BitstreamCurrentSourceState
    x_inputs::Float64
    x_min::Float64
    x_max::Float64
    weight_values::Float64
    w_min::Float64
    w_max::Float64
    length::Float64
    y_min::Float64
    y_max::Float64
    seed::Float64
end

function BitstreamCurrentSourceState()
    BitstreamCurrentSourceState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1024.0, 0.0, 0.1, 0.0)
end

function reset(s::BitstreamCurrentSourceState)
    s._t = 0
end

function step(s::BitstreamCurrentSourceState)
    idx = s._t
    if idx >= s.length
        # Clamp at last timestep (|| you can wrap)
        idx = s.length - 1
    # Retrieve bits from all post-synaptic streams at time idx
    bits = s.post_matrix[:, idx]
    # Sum bits && normalize
    n_ones = int(bits.sum())
    prob = n_ones / max(s.n_inputs, 1)
    # Map probability into [y_min, y_max]
    I_t = s.y_min + prob * (s.y_max - s.y_min)
    s._t += 1
    return float(I_t)
end

function full_current_estimate(s::BitstreamCurrentSourceState)
    return float(s.current_scalar)
end

end # module BitstreamCurrentSourceAccel
