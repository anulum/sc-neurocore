# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia retained clipped rational-recovery map

module SCClippedRationalRecoveryMapAccel

export simulate_trace

function simulate_trace(
    x0::Float64,
    y0::Float64,
    alpha::Float64,
    beta::Float64,
    j::Float64,
    x_threshold::Float64,
    clip_bound::Float64,
    n_steps::Int,
    current::Float64,
)
    values = (x0, y0, alpha, beta, j, x_threshold, clip_bound, current)
    all(isfinite, values) || throw(ArgumentError("SC rational-recovery inputs must be finite"))
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
    alpha > 0.0 && beta > 0.0 && clip_bound > 0.0 ||
        throw(ArgumentError("alpha, beta, and clip_bound must be positive"))
    abs(x0) <= clip_bound && abs(y0) <= clip_bound ||
        throw(ArgumentError("state exceeds clip_bound"))

    trace = Vector{Float64}(undef, n_steps)
    x = x0
    y = y0
    events = 0
    @inbounds for index in 1:n_steps
        x_previous = x
        field = x < 0.0 ? alpha * x : alpha * x / (1.0 + alpha * x)
        x_candidate = field + y + current + j
        y_candidate = y - beta * (x + 1.0)
        isfinite(x_candidate) && isfinite(y_candidate) ||
            throw(OverflowError("SC rational-recovery candidate became non-finite"))
        x = clamp(x_candidate, -clip_bound, clip_bound)
        y = clamp(y_candidate, -clip_bound, clip_bound)
        trace[index] = x
        if x >= x_threshold && x_previous < x_threshold
            events += 1
        end
    end
    return (trace = trace, events = events, xf = x, yf = y)
end

end
