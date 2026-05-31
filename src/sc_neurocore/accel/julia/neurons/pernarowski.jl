# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for Pernarowski

module PernarowskiAccel

export step!, simulate, PernarowskiNeuronState, validate

mutable struct PernarowskiNeuronState
    v::Float64
    w::Float64
    z::Float64
    alpha::Float64
    beta::Float64
    eps1::Float64
    eps2::Float64
    gamma::Float64
    dt::Float64
    v_threshold::Float64
end

function PernarowskiNeuronState()
    PernarowskiNeuronState(-1.0, 0.0, 0.0, 0.1, 0.5, 0.1, 0.001, 0.5, 0.1, 0.5)
end

function validate(s::PernarowskiNeuronState)::Bool
    isfinite(s.v) &&
    isfinite(s.w) &&
    isfinite(s.z) &&
    isfinite(s.alpha) &&
    isfinite(s.beta) &&
    isfinite(s.eps1) && s.eps1 > 0.0 &&
    isfinite(s.eps2) && s.eps2 > 0.0 &&
    isfinite(s.gamma) && s.gamma > 0.0 &&
    isfinite(s.dt) && s.dt > 0.0 &&
    isfinite(s.v_threshold)
end

function derivatives(s::PernarowskiNeuronState, v::Float64, w::Float64, z::Float64, current::Float64)
    if !all(isfinite, (v, w, z, current))
        return nothing
    end
    dv = v - v^3 / 3.0 - w - z + current
    dw = s.eps1 * (v - s.gamma * w + s.alpha)
    dz = s.eps2 * (s.beta * (v + 0.7) - z)
    all(isfinite, (dv, dw, dz)) ? (dv, dw, dz) : nothing
end

function rk4_candidate(s::PernarowskiNeuronState, current::Float64, dt::Float64)
    k1 = derivatives(s, s.v, s.w, s.z, current)
    k1 === nothing && return nothing
    k2 = derivatives(
        s,
        s.v + 0.5 * dt * k1[1],
        s.w + 0.5 * dt * k1[2],
        s.z + 0.5 * dt * k1[3],
        current,
    )
    k2 === nothing && return nothing
    k3 = derivatives(
        s,
        s.v + 0.5 * dt * k2[1],
        s.w + 0.5 * dt * k2[2],
        s.z + 0.5 * dt * k2[3],
        current,
    )
    k3 === nothing && return nothing
    k4 = derivatives(s, s.v + dt * k3[1], s.w + dt * k3[2], s.z + dt * k3[3], current)
    k4 === nothing && return nothing
    candidate = (
        s.v + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        s.w + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        s.z + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
    )
    all(isfinite, candidate) ? candidate : nothing
end

function step!(s::PernarowskiNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    if !validate(s) || !isfinite(I_ext) || !isfinite(dt) || dt <= 0.0
        return 0
    end

    v_prev = s.v
    candidate = rk4_candidate(s, I_ext, dt)
    candidate === nothing && return 0
    s.v, s.w, s.z = candidate
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = PernarowskiNeuronState()
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

end # module PernarowskiAccel
