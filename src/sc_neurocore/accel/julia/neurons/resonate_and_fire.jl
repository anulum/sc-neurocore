# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for resonate_and_fire

module ResonateAndFireAccel

export step!, simulate, ResonateAndFireNeuronState, valid, reset!, exact_flow

mutable struct ResonateAndFireNeuronState
    x::Float64
    y::Float64
    b::Float64
    omega::Float64
    threshold::Float64
    dt::Float64
end

function ResonateAndFireNeuronState()
    ResonateAndFireNeuronState(0.0, 0.0, -0.1, 1.0, 1.0, 0.05)
end

function valid(s::ResonateAndFireNeuronState)::Bool
    return isfinite(s.x) &&
        isfinite(s.y) &&
        isfinite(s.b) &&
        isfinite(s.omega) && s.omega > 0.0 &&
        isfinite(s.threshold) && s.threshold > 0.0 &&
        isfinite(s.dt) && s.dt > 0.0
end

function exact_flow(x::Float64, y::Float64, current::Float64, b::Float64, omega::Float64, dt::Float64)
    denominator = b * b + omega * omega
    damping_argument = b * dt
    angle = omega * dt
    if !isfinite(denominator) || denominator <= 0.0 || !isfinite(damping_argument) || !isfinite(angle)
        throw(DomainError((denominator, damping_argument, angle), "Resonate-and-fire exact-flow coefficients must be finite"))
    end

    x_ss = -b * current / denominator
    y_ss = omega * current / denominator
    decay = exp(damping_argument)
    cos_angle = cos(angle)
    sin_angle = sin(angle)
    if !all(isfinite, (x_ss, y_ss, decay, cos_angle, sin_angle))
        throw(DomainError((x_ss, y_ss, decay), "Resonate-and-fire exact-flow rotation must be finite"))
    end

    dx = x - x_ss
    dy = y - y_ss
    next_x = x_ss + decay * (dx * cos_angle - dy * sin_angle)
    next_y = y_ss + decay * (dx * sin_angle + dy * cos_angle)
    if !isfinite(next_x) || !isfinite(next_y)
        throw(DomainError((next_x, next_y), "Resonate-and-fire exact-flow update became non-finite"))
    end
    return next_x, next_y
end

function step!(s::ResonateAndFireNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)::Int
    if !isfinite(dt) || dt <= 0.0
        throw(DomainError(dt, "Resonate-and-fire timestep must be finite and positive"))
    end
    s.dt = dt
    if !isfinite(I_ext) || !valid(s)
        throw(DomainError((s.x, s.y, I_ext), "Resonate-and-fire state/current must be finite and well-formed"))
    end

    next_x, next_y = exact_flow(s.x, s.y, I_ext, s.b, s.omega, s.dt)
    radius = hypot(next_x, next_y)
    if !isfinite(radius)
        throw(DomainError((next_x, next_y, radius), "Resonate-and-fire exact-flow radius became non-finite"))
    end

    if radius >= s.threshold
        s.x = 0.0
        s.y = 0.0
        return 1
    end
    s.x = next_x
    s.y = next_y
    return 0
end

function reset!(s::ResonateAndFireNeuronState)::Nothing
    s.x = 0.0
    s.y = 0.0
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=2.0, dt::Float64=0.05)
    s = ResonateAndFireNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = hypot(s.x, s.y)
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module ResonateAndFireAccel
