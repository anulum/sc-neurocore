# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained upward-crossing Rulkov-map Julia backend

module SCUpwardCrossingRulkovMapAccel

export simulate_trace

"""
    simulate_trace(x0, y0, alpha, sigma, mu, x_threshold, n_steps, current)

Run the historical SC-NeuroCore Rulkov recurrence and return the fast-state
trace, configurable upward-crossing event count, and final state.
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
    values = (x0, y0, alpha, sigma, mu, x_threshold, current)
    all(isfinite, values) || throw(ArgumentError("state, parameters and current must be finite"))
    alpha > 0.0 || throw(ArgumentError("alpha must be positive"))
    mu > 0.0 || throw(ArgumentError("mu must be positive"))
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))

    trace = Vector{Float64}(undef, n_steps)
    x = x0
    y = y0
    events = 0
    for step in 1:n_steps
        x_previous = x
        boundary = alpha + y + current
        if x <= 0.0
            x_next = alpha / (1.0 - x) + y + current
        elseif x < boundary
            x_next = boundary
        else
            x_next = -1.0
        end
        y_next = y - mu * (x + 1.0) + mu * sigma
        isfinite(x_next) && isfinite(y_next) || throw(OverflowError("map candidate is non-finite"))
        x, y = x_next, y_next
        trace[step] = x
        events += Int(x >= x_threshold && x_previous < x_threshold)
    end
    return (trace = trace, events = events, xf = x, yf = y)
end

end
