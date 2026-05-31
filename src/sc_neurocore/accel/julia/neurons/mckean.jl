# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for McKean

module MckeanAccel

export step!, simulate, validate, McKeanNeuronState

mutable struct McKeanNeuronState
    v::Float64
    w::Float64
    a::Float64
    epsilon::Float64
    gamma::Float64
    dt::Float64
    v_peak::Float64
end

function McKeanNeuronState()
    McKeanNeuronState(0.0, 0.0, 0.25, 0.01, 0.5, 0.1, 0.8)
end

function validate(s::McKeanNeuronState, dt::Float64=s.dt)::Bool
    isfinite(s.v) &&
    isfinite(s.w) &&
    isfinite(s.a) && 0.0 < s.a < 1.0 &&
    isfinite(s.epsilon) && s.epsilon > 0.0 &&
    isfinite(s.gamma) && s.gamma > 0.0 &&
    isfinite(s.dt) && s.dt > 0.0 &&
    isfinite(dt) && dt > 0.0 &&
    isfinite(s.v_peak)
end

function _f(s::McKeanNeuronState, v::Float64)
    if !isfinite(v)
        return NaN
    end
    mid1 = s.a / 2.0
    mid2 = (1.0 + s.a) / 2.0
    if v < mid1
        return -v
    elseif v < mid2
        return v - s.a
    else
        return 1.0 - v
    end
end

function derivatives(s::McKeanNeuronState, v::Float64, w::Float64, current::Float64)
    if !all(isfinite, (v, w, current))
        return nothing
    end
    dv = _f(s, v) - w + current
    dw = s.epsilon * (v - s.gamma * w)
    all(isfinite, (dv, dw)) ? (dv, dw) : nothing
end

function rk4_candidate(s::McKeanNeuronState, current::Float64, dt::Float64)
    k1 = derivatives(s, s.v, s.w, current)
    k1 === nothing && return nothing
    k2 = derivatives(s, s.v + 0.5 * dt * k1[1], s.w + 0.5 * dt * k1[2], current)
    k2 === nothing && return nothing
    k3 = derivatives(s, s.v + 0.5 * dt * k2[1], s.w + 0.5 * dt * k2[2], current)
    k3 === nothing && return nothing
    k4 = derivatives(s, s.v + dt * k3[1], s.w + dt * k3[2], current)
    k4 === nothing && return nothing
    candidate = (
        s.v + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        s.w + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
    )
    all(isfinite, candidate) ? candidate : nothing
end

function step!(s::McKeanNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    if !validate(s, dt) || !isfinite(I_ext)
        return 0
    end

    v_prev = s.v
    candidate = rk4_candidate(s, I_ext, dt)
    candidate === nothing && return 0
    s.v, s.w = candidate
    return (s.v >= s.v_peak && v_prev < s.v_peak) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = McKeanNeuronState()
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

end # module MckeanAccel
