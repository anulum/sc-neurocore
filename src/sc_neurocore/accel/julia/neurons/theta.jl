# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for theta

module ThetaAccel

export step!, simulate, ThetaNeuronState, valid, reset!, wrap_phase

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

    s.dt = dt
    next_theta, spiked = _exact_candidate(s, I_ext)
    if !isfinite(next_theta)
        throw(DomainError(next_theta, "Theta exact-flow update became non-finite"))
    end

    s.theta = wrap_phase(next_theta)
    return spiked ? 1 : 0
end

function reset!(s::ThetaNeuronState)::Nothing
    s.theta = 0.0
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.01)
    s = ThetaNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.theta
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module ThetaAccel
