# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for morris_lecar

module MorrisLecarAccel

export step!, simulate, MorrisLecarNeuronState

mutable struct MorrisLecarNeuronState
    v::Float64
    w::Float64
    c_m::Float64
    g_ca::Float64
    g_k::Float64
    g_l::Float64
    e_ca::Float64
    e_k::Float64
    e_l::Float64
    v1::Float64
    v2::Float64
    v3::Float64
    v4::Float64
    phi::Float64
    dt::Float64
    v_threshold::Float64
end

function MorrisLecarNeuronState()
    MorrisLecarNeuronState(-60.0, 0.0, 20.0, 4.0, 8.0, 2.0, 120.0, -84.0, -60.0, -1.2, 18.0, 12.0, 17.4, 1.0 / 15.0, 0.1, 0.0)
end

function _m_inf(s::MorrisLecarNeuronState, v)
    return 0.5 * (1.0 + tanh((v - s.v1) / s.v2))
end

function _w_inf(s::MorrisLecarNeuronState, v)
    return 0.5 * (1.0 + tanh((v - s.v3) / s.v4))
end

function _lam(s::MorrisLecarNeuronState, v)
    return s.phi * cosh((v - s.v3) / (2.0 * s.v4))
end

function _valid(s::MorrisLecarNeuronState)
    values = (
        s.v, s.w, s.c_m, s.g_ca, s.g_k, s.g_l, s.e_ca, s.e_k, s.e_l,
        s.v1, s.v2, s.v3, s.v4, s.phi, s.dt, s.v_threshold
    )
    return all(isfinite, values) &&
        s.c_m > 0.0 &&
        s.g_ca > 0.0 &&
        s.g_k > 0.0 &&
        s.g_l > 0.0 &&
        s.v2 > 0.0 &&
        s.v4 > 0.0 &&
        s.phi > 0.0 &&
        s.dt > 0.0 &&
        0.0 <= s.w <= 1.0
end

function step!(s::MorrisLecarNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    if !_valid(s) || !isfinite(I_ext) || !isfinite(dt) || dt <= 0.0
        return -1
    end

    v_prev = s.v
    m_inf = _m_inf(s, s.v)
    w_inf = _w_inf(s, s.v)
    lam = _lam(s, s.v)
    i_ca = s.g_ca * m_inf * (s.v - s.e_ca)
    i_k = s.g_k * s.w * (s.v - s.e_k)
    i_l = s.g_l * (s.v - s.e_l)
    candidate = MorrisLecarNeuronState(
        s.v + (-i_ca - i_k - i_l + I_ext) / s.c_m * dt,
        s.w + lam * (w_inf - s.w) * dt,
        s.c_m, s.g_ca, s.g_k, s.g_l, s.e_ca, s.e_k, s.e_l,
        s.v1, s.v2, s.v3, s.v4, s.phi, dt, s.v_threshold,
    )
    if !_valid(candidate)
        return -1
    end
    s.v = candidate.v
    s.w = candidate.w
    s.dt = candidate.dt
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = MorrisLecarNeuronState()
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

end # module MorrisLecarAccel
