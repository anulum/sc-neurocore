# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for fitzhugh_rinzel

module FitzhughRinzelAccel

export step!, simulate, FitzHughRinzelNeuronState

mutable struct FitzHughRinzelNeuronState
    v::Float64
    w::Float64
    y::Float64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    delta::Float64
    mu::Float64
    dt::Float64
    v_threshold::Float64
end

function FitzHughRinzelNeuronState()
    FitzHughRinzelNeuronState(-1.0, -0.5, 0.0, 0.7, 0.8, -0.775, 1.0, 0.08, 0.0001, 0.1, 1.0)
end

function _valid(s::FitzHughRinzelNeuronState, dt::Float64)::Bool
    return all(isfinite, (s.v, s.w, s.y, s.a, s.b, s.c, s.d, s.delta, s.mu, s.v_threshold, dt)) &&
        s.delta > 0.0 && s.mu > 0.0 && dt > 0.0
end

function _derivatives(s::FitzHughRinzelNeuronState, I_ext::Float64)
    if !isfinite(I_ext)
        return nothing
    end
    dv = s.v - s.v^3 / 3.0 - s.w + s.y + I_ext
    dw = s.delta * (s.a + s.v - s.b * s.w)
    dy = s.mu * (s.c - s.v - s.d * s.y)
    if all(isfinite, (dv, dw, dy))
        return (dv, dw, dy)
    end
    return nothing
end

function step!(s::FitzHughRinzelNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    if !_valid(s, dt)
        return 0
    end
    v_prev = s.v
    derivatives = _derivatives(s, I_ext)
    if derivatives === nothing
        return 0
    end
    dv, dw, dy = derivatives
    next_v = s.v + dv * dt
    next_w = s.w + dw * dt
    next_y = s.y + dy * dt
    if !all(isfinite, (next_v, next_w, next_y))
        return 0
    end
    s.v = next_v
    s.w = next_w
    s.y = next_y
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = FitzHughRinzelNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module FitzhughRinzelAccel
