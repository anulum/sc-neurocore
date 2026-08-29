# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Ibarz analysis profile of the Shilnikov-Rulkov map

module IbarzTanakaMapAccel

export simulate_trace

"""
    simulate_trace(v0, u0, alpha, mu, sigma, n_steps, current)

Run the Shilnikov-Rulkov (2004) map as profiled by Ibarz et al. (2007),
Eqs. 2-3, and return
`(trace, events, vf, uf)`. The trace stores the fast state after each
simultaneous map iteration; events count executions of the reset branch.
"""
function simulate_trace(
    v0::Float64,
    u0::Float64,
    alpha::Float64,
    mu::Float64,
    sigma::Float64,
    n_steps::Int,
    current::Float64,
)
    values = (v0, u0, alpha, mu, sigma, current)
    all(isfinite, values) || throw(ArgumentError("state, parameters and current must be finite"))
    alpha > 0.0 || throw(ArgumentError("alpha must be positive"))
    mu > 0.0 || throw(ArgumentError("mu must be positive"))
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))

    trace = Vector{Float64}(undef, n_steps)
    v = v0
    u = u0
    events = 0
    for step in 1:n_steps
        lower = -1.0 - alpha / 2.0
        upper = 1.0 + current + u
        if v < lower
            v_next = -(alpha * alpha) / 4.0 - alpha + current + u
        elseif v <= 0.0
            v_next = alpha * v + (v + 1.0) * (v + 1.0) + current + u
        elseif v < upper
            v_next = upper
        else
            v_next = -1.0
            events += 1
        end
        u_next = u - mu * (v + 1.0 - sigma)
        isfinite(v_next) && isfinite(u_next) || throw(OverflowError("map candidate is non-finite"))
        v, u = v_next, u_next
        trace[step] = v
    end
    return (trace = trace, events = events, vf = v, uf = u)
end

end
