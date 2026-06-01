# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for traub_miles

module TraubMilesAccel

export step!, simulate, validate, TraubMilesNeuronState

mutable struct TraubMilesNeuronState
    v::Float64
    m::Float64
    h::Float64
    n::Float64
    g_na::Float64
    g_k::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    dt::Float64
    v_threshold::Float64
end

function TraubMilesNeuronState()
    TraubMilesNeuronState(-67.0, 0.05, 0.6, 0.3, 100.0, 80.0, 0.1, 50.0, -100.0, -67.0, 0.01, -20.0)
end

finite_gate(x::Float64)::Bool = isfinite(x) && 0.0 <= x <= 1.0

function validate(s::TraubMilesNeuronState)::Bool
    return isfinite(s.v) &&
        finite_gate(s.m) &&
        finite_gate(s.h) &&
        finite_gate(s.n) &&
        isfinite(s.g_na) &&
        s.g_na >= 0.0 &&
        isfinite(s.g_k) &&
        s.g_k >= 0.0 &&
        isfinite(s.g_l) &&
        s.g_l >= 0.0 &&
        isfinite(s.e_na) &&
        isfinite(s.e_k) &&
        isfinite(s.e_l) &&
        isfinite(s.dt) &&
        s.dt > 0.0 &&
        isfinite(s.v_threshold)
end

function rates(v::Float64)
    d = v + 54.0
    am = (abs(d) > 1e-06) ? 0.32 * d / (1.0 - exp(-d / 4.0)) : 8.0
    d2 = v + 27.0
    bm = (abs(d2) > 1e-06) ? 0.28 * d2 / (exp(d2 / 5.0) - 1.0) : 5.6
    ah = 0.128 * exp(-(v + 50.0) / 18.0)
    bh = 4.0 / (1.0 + exp(-(v + 27.0) / 5.0))
    d3 = v + 52.0
    an = (abs(d3) > 1e-06) ? 0.032 * d3 / (1.0 - exp(-d3 / 5.0)) : 0.32
    bn = 0.5 * exp(-(v + 57.0) / 40.0)
    return am, bm, ah, bh, an, bn
end

function derivatives(s::TraubMilesNeuronState, v::Float64, m::Float64, h::Float64, n::Float64, I_ext::Float64)
    finite_gate(m) && finite_gate(h) && finite_gate(n) && isfinite(v) || return nothing
    am, bm, ah, bh, an, bn = rates(v)
    if !all(isfinite, (am, bm, ah, bh, an, bn)) || any(x -> x < 0.0, (am, bm, ah, bh, an, bn))
        return nothing
    end
    dm = am * (1.0 - m) - bm * m
    dh = ah * (1.0 - h) - bh * h
    dn = an * (1.0 - n) - bn * n
    i_na = s.g_na * m ^ 3 * h * (v - s.e_na)
    i_k = s.g_k * n ^ 4 * (v - s.e_k)
    i_l = s.g_l * (v - s.e_l)
    dv = -i_na - i_k - i_l + I_ext
    all(isfinite, (dv, dm, dh, dn, i_na, i_k, i_l)) || return nothing
    return dv, dm, dh, dn
end

function rk4_substep(s::TraubMilesNeuronState, v::Float64, m::Float64, h::Float64, n::Float64, I_ext::Float64)
    k1 = derivatives(s, v, m, h, n, I_ext)
    k1 === nothing && return nothing
    k2 = derivatives(
        s,
        v + 0.5 * s.dt * k1[1],
        m + 0.5 * s.dt * k1[2],
        h + 0.5 * s.dt * k1[3],
        n + 0.5 * s.dt * k1[4],
        I_ext,
    )
    k2 === nothing && return nothing
    k3 = derivatives(
        s,
        v + 0.5 * s.dt * k2[1],
        m + 0.5 * s.dt * k2[2],
        h + 0.5 * s.dt * k2[3],
        n + 0.5 * s.dt * k2[4],
        I_ext,
    )
    k3 === nothing && return nothing
    k4 = derivatives(
        s,
        v + s.dt * k3[1],
        m + s.dt * k3[2],
        h + s.dt * k3[3],
        n + s.dt * k3[4],
        I_ext,
    )
    k4 === nothing && return nothing
    next_v = v + s.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
    next_m = m + s.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
    next_h = h + s.dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
    next_n = n + s.dt * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) / 6.0
    isfinite(next_v) && finite_gate(next_m) && finite_gate(next_h) && finite_gate(next_n) || return nothing
    return next_v, next_m, next_h, next_n
end

function step!(s::TraubMilesNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !validate(s) || !isfinite(I_ext)
        return -1
    end

    v_prev = s.v
    v = s.v
    m = s.m
    h = s.h
    n = s.n
    for _ in 1:10
        candidate = rk4_substep(s, v, m, h, n, I_ext)
        candidate === nothing && return -1
        next_v, next_m, next_h, next_n = candidate
        v = next_v
        m = next_m
        h = next_h
        n = next_n
    end
    s.v = v
    s.m = m
    s.h = h
    s.n = n
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = TraubMilesNeuronState()
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

end # module TraubMilesAccel
