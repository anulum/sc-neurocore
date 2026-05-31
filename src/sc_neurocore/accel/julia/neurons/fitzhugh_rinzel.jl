# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for fitzhugh_rinzel

module FitzhughRinzelAccel

export step!, simulate, validate_state, FitzHughRinzelNeuronState

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

function validate_state(s::FitzHughRinzelNeuronState)
    return all(isfinite, (s.v, s.w, s.y, s.a, s.b, s.c, s.d, s.delta, s.mu, s.dt, s.v_threshold)) &&
        s.b > 0.0 && s.d > 0.0 && s.delta > 0.0 && s.mu > 0.0 && s.dt > 0.0
end

function _derivatives(s::FitzHughRinzelNeuronState, v::Float64, w::Float64, y::Float64, i_ext::Float64)
    if !all(isfinite, (v, w, y, i_ext))
        return nothing
    end
    dv = v - v^3 / 3.0 - w + y + i_ext
    dw = s.delta * (s.a + v - s.b * w)
    dy = s.mu * (s.c - v - s.d * y)
    if all(isfinite, (dv, dw, dy))
        return (dv, dw, dy)
    end
    return nothing
end

function _rk4_candidate(s::FitzHughRinzelNeuronState, i_ext::Float64)
    dt = s.dt
    v0 = s.v
    w0 = s.w
    y0 = s.y
    k1 = _derivatives(s, v0, w0, y0, i_ext)
    k1 === nothing && return nothing
    k2 = _derivatives(s, v0 + 0.5 * dt * k1[1], w0 + 0.5 * dt * k1[2], y0 + 0.5 * dt * k1[3], i_ext)
    k2 === nothing && return nothing
    k3 = _derivatives(s, v0 + 0.5 * dt * k2[1], w0 + 0.5 * dt * k2[2], y0 + 0.5 * dt * k2[3], i_ext)
    k3 === nothing && return nothing
    k4 = _derivatives(s, v0 + dt * k3[1], w0 + dt * k3[2], y0 + dt * k3[3], i_ext)
    k4 === nothing && return nothing
    return (
        v0 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        w0 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        y0 + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
    )
end

function step!(s::FitzHughRinzelNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    if !isfinite(dt) || dt <= 0.0
        return 0
    end
    old_dt = s.dt
    s.dt = dt
    if !(isfinite(I_ext) && validate_state(s))
        s.dt = old_dt
        return 0
    end
    v_prev = s.v
    candidate = _rk4_candidate(s, I_ext)
    if candidate === nothing || !all(isfinite, candidate)
        s.dt = old_dt
        return 0
    end
    s.v, s.w, s.y = candidate
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
