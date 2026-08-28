# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Cazelles four-branch map

module CazellesMapAccel

export simulate_trace

function simulate_trace(
    x::Float64,
    alpha::Float64,
    x0::Float64,
    x1::Float64,
    x2::Float64,
    x3::Float64,
    x4::Float64,
    a1::Float64,
    a2::Float64,
    a3::Float64,
    a4::Float64,
    b1::Float64,
    b2::Float64,
    b3::Float64,
    b4::Float64,
    exponent::Int,
    n_steps::Int,
    current::Float64,
)
    values = (x, alpha, x0, x1, x2, x3, x4, a1, a2, a3, a4, b1, b2, b3, b4, current)
    all(isfinite, values) || throw(ArgumentError("Cazelles inputs must be finite"))
    0.0 <= alpha < 1.0 || throw(ArgumentError("alpha must satisfy 0 <= alpha < 1"))
    exponent in (1, 2) || throw(ArgumentError("exponent must be 1 or 2"))
    x0 < x1 < x2 < x3 < x4 || throw(ArgumentError("branch bounds must increase"))
    x0 <= x <= x4 || throw(ArgumentError("x lies outside the map domain"))
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))

    trace = Vector{Float64}(undef, n_steps)
    events = 0
    for index in 1:n_steps
        base = if x < x1
            a1 + b1 * x
        elseif x < x2
            a2 + b2 * x
        elseif x < x3
            a3 + b3 * x
        else
            a4 + b4 * x
        end
        power = exponent == 1 ? x : x * x
        candidate = base + alpha * power + current
        isfinite(candidate) || throw(DomainError(candidate, "non-finite Cazelles candidate"))
        tolerance = 8.0 * eps(max(1.0, abs(x0), abs(x4)))
        if x0 - tolerance <= candidate < x0
            candidate = x0
        elseif x4 < candidate <= x4 + tolerance
            candidate = x4
        end
        x0 <= candidate <= x4 || throw(DomainError(candidate, "Cazelles candidate left domain"))
        events += x >= x1 && candidate < x1 ? 1 : 0
        x = candidate
        trace[index] = x
    end
    return (trace = trace, events = events, xf = x)
end

end # module CazellesMapAccel
