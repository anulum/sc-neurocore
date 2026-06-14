# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for the Cazelles 2001 bursting map

# Parity contract: `simulate_trace` reproduces
# `sc_neurocore.neurons.models.cazelles_map.CazellesMapNeuron.simulate`
# bit-for-bit — the map is exact floating-point arithmetic (a*x*(1-x),
# additions, a clamp), so identical operation order yields identical results.
#
# Reference: Cazelles, B., Courbage, M. & Rabinovich, M. (2001).
# Europhys. Lett. 56(4):504-509.

module CazellesMapAccel

export simulate_trace

"""
    simulate_trace(x0, y0, a, epsilon, sigma, x_threshold, n_steps, current)

Run `n_steps` of the Cazelles bursting map from state `(x0, y0)` under a
constant input `current`. Returns a named tuple `(trace, spikes, xf, yf)`
where `trace[t]` is `x` after step `t`, `spikes` counts threshold crossings,
and `(xf, yf)` is the final state.
"""
function simulate_trace(
    x0::Float64,
    y0::Float64,
    a::Float64,
    epsilon::Float64,
    sigma::Float64,
    x_threshold::Float64,
    n_steps::Int,
    current::Float64,
)
    trace = Vector{Float64}(undef, n_steps)
    x = x0
    y = y0
    spikes = 0
    for t in 1:n_steps
        f = a * x * (1.0 - x)
        x_new = f - y + current
        y_new = y + epsilon * (x - sigma)
        x = min(2.0, max(-2.0, x_new))
        y = y_new
        trace[t] = x
        if x >= x_threshold
            spikes += 1
        end
    end
    return (trace = trace, spikes = spikes, xf = x, yf = y)
end

end # module CazellesMapAccel
