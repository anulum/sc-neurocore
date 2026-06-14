# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for the Medvedev 2005 1D spiking map

# Parity contract: `simulate_trace` reproduces
# `sc_neurocore.neurons.models.medvedev_map.MedvedevMapNeuron.simulate`
# bit-for-bit — the map is exact floating-point arithmetic (a multiply, an add,
# and a fold into [0, 1)). The fold uses `mod(x, 1.0)` (floored remainder),
# which equals Python's `x % 1.0` and Rust's `rem_euclid(1.0)` bit-for-bit; note
# Julia's `%` operator is `rem` (truncated) and must NOT be used here.
#
# Reference: Medvedev, G.S. (2005). Physica D 202:37-59.

module MedvedevMapAccel

export simulate_trace

"""
    simulate_trace(x0, alpha, beta, x_threshold, n_steps, current)

Run `n_steps` of the Medvedev 1D map from state `x0` under a constant input
`current`. Returns a named tuple `(trace, spikes, xf)` where `trace[t]` is `x`
after step `t` (folded into [0, 1)), `spikes` counts upward crossings of
`x_threshold`, and `xf` is the final state.
"""
function simulate_trace(
    x0::Float64,
    alpha::Float64,
    beta::Float64,
    x_threshold::Float64,
    n_steps::Int,
    current::Float64,
)
    trace = Vector{Float64}(undef, n_steps)
    x = x0
    spikes = 0
    for t in 1:n_steps
        x_prev = x
        if x < beta
            x = alpha * x + current
        else
            x = alpha * (1.0 - x) + current
        end
        x = mod(x, 1.0)
        trace[t] = x
        if x >= x_threshold && x_prev < x_threshold
            spikes += 1
        end
    end
    return (trace = trace, spikes = spikes, xf = x)
end

end # module MedvedevMapAccel
