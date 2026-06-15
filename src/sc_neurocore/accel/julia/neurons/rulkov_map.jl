# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for the Rulkov 2001 fast/slow map

# Parity contract: `simulate_trace` reproduces
# `sc_neurocore.neurons.models.rulkov_map.RulkovMapNeuron.simulate` bit-for-bit
# — the fast map is exact floating-point arithmetic (one division, additions,
# multiplications, no transcendental functions), so identical operation order
# yields an identical trace, spike count and final state.
#
# Reference: Rulkov, N.F. (2002). Phys. Rev. E 65:041922.

module RulkovMapAccel

export simulate_trace

"""
    simulate_trace(x0, y0, alpha, sigma, mu, x_threshold, n_steps, current)

Run `n_steps` of the Rulkov fast/slow map from state `(x0, y0)` under a
constant input `current`. Returns a named tuple `(trace, spikes, xf, yf)`
where `trace[t]` is `x` after step `t`, `spikes` counts upward crossings of
`x_threshold`, and `(xf, yf)` is the final state.
"""
function simulate_trace(
    x0::Float64,
    y0::Float64,
    alpha::Float64,
    sigma::Float64,
    mu::Float64,
    x_threshold::Float64,
    n_steps::Int,
    current::Float64,
)
    trace = Vector{Float64}(undef, n_steps)
    x = x0
    y = y0
    spikes = 0
    for t in 1:n_steps
        x_prev = x
        branch_boundary = alpha + y + current
        if x <= 0
            x_new = alpha / (1.0 - x) + y + current
        elseif x < branch_boundary
            x_new = branch_boundary
        else
            x_new = -1.0
        end
        y_new = y - mu * (x + 1.0) + mu * sigma
        x = x_new
        y = y_new
        trace[t] = x
        if x >= x_threshold && x_prev < x_threshold
            spikes += 1
        end
    end
    return (trace = trace, spikes = spikes, xf = x, yf = y)
end

end # module RulkovMapAccel
