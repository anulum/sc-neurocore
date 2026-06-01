# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Sherman-Rinzel-Keizer RK4 mirror

module ShermanRinzelKeizerAccel

export step!, simulate, validate_state, ShermanRinzelKeizerNeuronState

const TAU_N = 9.09

mutable struct ShermanRinzelKeizerNeuronState
    v::Float64
    n::Float64
    s::Float64
    g_ca::Float64
    g_k::Float64
    g_s::Float64
    e_ca::Float64
    e_k::Float64
    tau_s::Float64
    dt::Float64
    v_threshold::Float64
end

function ShermanRinzelKeizerNeuronState()
    ShermanRinzelKeizerNeuronState(-50.0, 0.1, 0.1, 3.6, 10.0, 4.0, 25.0, -75.0, 5000.0, 0.5, -20.0)
end

_gate(x::Float64)::Bool = isfinite(x) && 0.0 <= x <= 1.0
_sigmoid(x::Float64)::Float64 = 1.0 / (1.0 + exp(-clamp(x, -80.0, 80.0)))

function validate_state(s::ShermanRinzelKeizerNeuronState)::Bool
    return isfinite(s.v) && -200.0 <= s.v <= 200.0 && _gate(s.n) && _gate(s.s) &&
        isfinite(s.g_ca) && s.g_ca > 0.0 &&
        isfinite(s.g_k) && s.g_k > 0.0 &&
        isfinite(s.g_s) && s.g_s >= 0.0 &&
        isfinite(s.e_ca) && isfinite(s.e_k) &&
        isfinite(s.tau_s) && s.tau_s > 0.0 &&
        isfinite(s.dt) && s.dt > 0.0 && isfinite(s.v_threshold)
end

function _derivatives(s::ShermanRinzelKeizerNeuronState, v::Float64, n_gate::Float64, s_gate::Float64, current::Float64)
    if !(isfinite(v) && isfinite(n_gate) && isfinite(s_gate) && isfinite(current))
        return nothing
    end
    m_inf = _sigmoid((v + 20.0) / 12.0)
    n_inf = _sigmoid((v + 16.0) / 5.0)
    s_inf = _sigmoid((v + 35.0) / 10.0)
    i_ca = s.g_ca * m_inf * (v - s.e_ca)
    i_k = s.g_k * n_gate * (v - s.e_k)
    i_s = s.g_s * s_gate * (v - s.e_k)
    dv = -i_ca - i_k - i_s + current
    dn = (n_inf - n_gate) / TAU_N
    ds = (s_inf - s_gate) / s.tau_s
    return (isfinite(dv) && isfinite(dn) && isfinite(ds)) ? (dv, dn, ds) : nothing
end

function _rk4_candidate(s::ShermanRinzelKeizerNeuronState, current::Float64)
    half_dt = 0.5 * s.dt
    k1 = _derivatives(s, s.v, s.n, s.s, current)
    k1 === nothing && return nothing
    k2 = _derivatives(s, s.v + half_dt * k1[1], s.n + half_dt * k1[2], s.s + half_dt * k1[3], current)
    k2 === nothing && return nothing
    k3 = _derivatives(s, s.v + half_dt * k2[1], s.n + half_dt * k2[2], s.s + half_dt * k2[3], current)
    k3 === nothing && return nothing
    k4 = _derivatives(s, s.v + s.dt * k3[1], s.n + s.dt * k3[2], s.s + s.dt * k3[3], current)
    k4 === nothing && return nothing
    next_v = s.v + s.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
    next_n = s.n + s.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
    next_s = s.s + s.dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
    if !(isfinite(next_v) && -200.0 <= next_v <= 200.0 && _gate(next_n) && _gate(next_s))
        return nothing
    end
    return next_v, next_n, next_s
end

function step!(s::ShermanRinzelKeizerNeuronState, current::Float64=0.0; dt::Float64=s.dt)
    s.dt = dt
    if !(validate_state(s) && isfinite(current))
        return 0
    end
    v_prev = s.v
    candidate = _rk4_candidate(s, current)
    candidate === nothing && return 0
    s.v, s.n, s.s = candidate
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.5)
    s = ShermanRinzelKeizerNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v
        spikes += result > 0 ? 1 : 0
    end
    return trace, spikes
end

end
