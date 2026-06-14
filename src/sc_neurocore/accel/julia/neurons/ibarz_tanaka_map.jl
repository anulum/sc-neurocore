# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for the Ibarz-Tanaka piecewise-linear map

# Parity contract: `simulate_trace` reproduces
# `sc_neurocore.neurons.models.ibarz_tanaka_map.IbarzTanakaMapNeuron.simulate`
# bit-for-bit — the map is exact floating-point arithmetic (one division,
# additions, multiplications, no transcendental functions), so identical
# operation order yields an identical trace, spike count and final state.
#
# Reference: Ibarz, B., Casado, J.M. & Sanjuán, M.A.F. (2011).
# Phys. Rep. 501:1-74.

module IbarzTanakaMapAccel

export simulate_trace

"""
    simulate_trace(x0, y0, alpha, beta, mu, sigma, x_threshold, x_reset, n_steps, current)

Run `n_steps` of the Ibarz-Tanaka map from state `(x0, y0)` under a constant
input `current`. Returns a named tuple `(trace, spikes, xf, yf)` where
`trace[t]` is `x` after step `t` (already reset to `x_reset` on a spiking step),
`spikes` counts threshold crossings, and `(xf, yf)` is the final state.
"""
function simulate_trace(
    x0::Float64,
    y0::Float64,
    alpha::Float64,
    beta::Float64,
    mu::Float64,
    sigma::Float64,
    x_threshold::Float64,
    x_reset::Float64,
    n_steps::Int,
    current::Float64,
)
    trace = Vector{Float64}(undef, n_steps)
    x = x0
    y = y0
    spikes = 0
    for t in 1:n_steps
        if x <= 0.0
            f = alpha / (1.0 - x)
        else
            f = alpha + beta * x
        end
        x_new = f + y + current
        y_new = y - mu * (x + 1.0) + mu * sigma
        y = y_new
        if x_new >= x_threshold
            x = x_reset
            spikes += 1
        else
            x = x_new
        end
        trace[t] = x
    end
    return (trace = trace, spikes = spikes, xf = x, yf = y)
end

end # module IbarzTanakaMapAccel
