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

function step!(s::ThetaNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)::Int
    s.dt = dt
    if !isfinite(I_ext) || !valid(s)
        throw(DomainError((s.theta, s.dt, I_ext), "Theta state/current must be finite with positive dt"))
    end

    theta_prev = s.theta
    cos_theta = cos(s.theta)
    dtheta = ((1.0 - cos_theta) + (1.0 + cos_theta) * I_ext) * s.dt
    next_theta = s.theta + dtheta
    if !isfinite(dtheta) || !isfinite(next_theta)
        throw(DomainError((dtheta, next_theta), "Theta phase increment must remain finite"))
    end

    spike = (theta_prev < pi * 0.99 && next_theta >= pi * 0.99) ? 1 : 0
    s.theta = wrap_phase(next_theta)
    return spike
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
