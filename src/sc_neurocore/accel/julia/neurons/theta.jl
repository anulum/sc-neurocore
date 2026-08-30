# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for theta

module ThetaAccel

export step!, simulate, simulate_trace, simulate_complete, ThetaNeuronState, valid, reset!, wrap_phase

mutable struct ThetaNeuronState
    theta::Float64
    dt::Float64
end

function ThetaNeuronState()
    ThetaNeuronState(0.0, 0.01)
end

function valid(s::ThetaNeuronState)::Bool
    return isfinite(s.theta) && isfinite(s.dt) && s.dt > 0.0
end

function wrap_phase(theta::Float64)::Float64
    return mod(theta + pi, 2.0 * pi) - pi
end

function event_packet_representable(s::ThetaNeuronState, I_ext::Float64)::Bool
    return I_ext <= 0.0 || sqrt(I_ext) * s.dt <= pi
end

function _exact_candidate(s::ThetaNeuronState, I_ext::Float64)
    y = tan(s.theta / 2.0)
    if I_ext > 0.0
        root_i = sqrt(I_ext)
        phase = atan(y / root_i)
        next_phase = phase + root_i * s.dt
        if abs(cos(next_phase)) <= 1.0e-15
            return -pi, next_phase >= pi / 2.0
        end
        return wrap_phase(2.0 * atan(root_i * tan(next_phase))), next_phase >= pi / 2.0
    elseif I_ext == 0.0
        denominator = 1.0 - y * s.dt
        if abs(denominator) <= 1.0e-15
            return -pi, true
        end
        return wrap_phase(2.0 * atan(y / denominator)), denominator <= 0.0
    end

    root_i = sqrt(-I_ext)
    if abs(y + root_i) <= 1.0e-15
        return s.theta, false
    end
    ratio = (y - root_i) / (y + root_i)
    evolved = ratio * exp(2.0 * root_i * s.dt)
    denominator = 1.0 - evolved
    spiked = (ratio < 1.0 && evolved >= 1.0) || abs(denominator) <= 1.0e-15
    if spiked && abs(denominator) <= 1.0e-15
        return -pi, true
    end
    return wrap_phase(2.0 * atan(root_i * (1.0 + evolved) / denominator)), spiked
end

function step!(s::ThetaNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)::Int
    if !all(isfinite, (s.theta, dt, I_ext)) || dt <= 0.0
        throw(DomainError((s.theta, s.dt, I_ext), "Theta state/current must be finite with positive dt"))
    end

    candidate_state = ThetaNeuronState(s.theta, dt)
    if !event_packet_representable(candidate_state, I_ext)
        throw(DomainError((dt, I_ext), "Theta step can contain more than one source event"))
    end
    next_theta, spiked = _exact_candidate(candidate_state, I_ext)
    if !isfinite(next_theta)
        throw(DomainError(next_theta, "Theta exact-flow update became non-finite"))
    end

    s.dt = dt
    s.theta = wrap_phase(next_theta)
    return spiked ? 1 : 0
end

function reset!(s::ThetaNeuronState)::Nothing
    s.theta = 0.0
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.01)
    result = simulate_trace(0.0, dt, n_steps, I_ext)
    return result.trace, result.spikes
end

function simulate_trace(
    theta::Float64,
    dt::Float64,
    n_steps::Int,
    I_ext::Float64,
)
    if n_steps < 0
        throw(ArgumentError("Theta n_steps must be non-negative"))
    end
    if !all(isfinite, (theta, dt, I_ext)) || dt <= 0.0
        throw(DomainError((theta, dt, I_ext), "Theta state/current must be finite with positive dt"))
    end
    packet = simulate_complete(theta, dt, n_steps, I_ext)
    return (trace=packet.trace, spikes=sum(packet.events), thetaf=packet.thetaf)
end

function simulate_complete(
    theta::Float64,
    dt::Float64,
    n_steps::Int,
    I_ext::Float64,
)
    if n_steps < 0
        throw(ArgumentError("Theta n_steps must be non-negative"))
    end
    if !all(isfinite, (theta, dt, I_ext)) || dt <= 0.0
        throw(DomainError((theta, dt, I_ext), "Theta state/current must be finite with positive dt"))
    end
    s = ThetaNeuronState(wrap_phase(theta), dt)
    if !event_packet_representable(s, I_ext)
        throw(DomainError((dt, I_ext), "Theta step can contain more than one source event"))
    end
    trace = zeros(n_steps)
    events = zeros(UInt8, n_steps)
    for t in 1:n_steps
        events[t] = UInt8(step!(s, I_ext))
        trace[t] = s.theta
    end
    return (trace=trace, events=events, thetaf=s.theta)
end

end # module ThetaAccel
