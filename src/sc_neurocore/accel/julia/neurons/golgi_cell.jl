# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore Julia accelerator for golgi_cell

module GolgiCellAccel

export step!, simulate, GolgiCellState

mutable struct GolgiCellState
    v::Float64
    m::Float64
    h::Float64
    p_na::Float64
    n::Float64
    a::Float64
    b::Float64
    w::Float64
    m_t::Float64
    s::Float64
    c_n::Float64
    r::Float64
    ca::Float64
    g_na_t::Float64
    g_na_p::Float64
    g_kdr::Float64
    g_ka::Float64
    g_km::Float64
    g_cat::Float64
    g_can::Float64
    g_bk::Float64
    g_sk::Float64
    g_h::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_ca::Float64
    e_h::Float64
    e_l::Float64
    c_m::Float64
    tau_ca::Float64
    kd_bk::Float64
    kd_sk::Float64
    dt::Float64
    sub_steps::Int
    gain::Float64
end

function GolgiCellState()
    return GolgiCellState(-60.0, 0.02, 0.85, 0.01, 0.05, 0.1, 0.8, 0.01, 0.01, 0.9, 0.01, 0.1, 0.05, 48.0, 0.2, 16.0, 8.0, 1.0, 0.5, 1.0, 3.0, 1.0, 0.1, 0.05, 55.0, -90.0, 120.0, -40.0, -55.0, 1.0, 200.0, 1.0, 0.5, 0.5, 10, 1.0)
end

function _safe_rate(a::Float64, vhalf::Float64, v::Float64, k::Float64, fallback::Float64)
    d = v + vhalf
    abs(d) < 1.0e-7 && return fallback
    return a * d / (1.0 - exp(-d / k))
end

function _boltz(v::Float64, vh::Float64, k::Float64)
    x = (v - vh) / k
    if x >= 0.0
        return 1.0 / (1.0 + exp(-x))
    end
    ex = exp(x)
    return ex / (1.0 + ex)
end

_voltage(value::Float64) = isfinite(value) && -100.0 <= value <= 60.0
_probability(value::Float64) = isfinite(value) && 0.0 <= value <= 1.0

function _gate_alpha_beta(previous::Float64, alpha::Float64, beta::Float64, phi::Float64, dt::Float64)
    total = phi * (alpha + beta)
    if !all(isfinite, (previous, alpha, beta, total, dt)) || total <= 0.0
        return nothing
    end
    steady = alpha / (alpha + beta)
    return min(1.0, max(0.0, steady + (previous - steady) * exp(-total * dt)))
end

function _gate_inf(previous::Float64, steady::Float64, tau::Float64, dt::Float64)
    if !all(isfinite, (previous, steady, tau, dt)) || tau <= 0.0
        return nothing
    end
    return min(1.0, max(0.0, steady + (previous - steady) * exp(-dt / tau)))
end

function _calcium(previous::Float64, entry::Float64, tau::Float64, dt::Float64)
    if !all(isfinite, (previous, entry, tau, dt)) || tau <= 0.0 || previous < 0.0
        return nothing
    end
    steady = entry * tau
    value = steady + (previous - steady) * exp(-dt / tau)
    isfinite(value) || return nothing
    return max(0.0, value)
end

function _valid_state(s::GolgiCellState)
    gates = (s.m, s.h, s.p_na, s.n, s.a, s.b, s.w, s.m_t, s.s, s.c_n, s.r)
    conductances = (s.g_na_t, s.g_na_p, s.g_kdr, s.g_ka, s.g_km, s.g_cat, s.g_can, s.g_bk, s.g_sk, s.g_h, s.g_l)
    return _voltage(s.v) &&
        all(_probability, gates) &&
        all(g -> isfinite(g) && g >= 0.0, conductances) &&
        all(isfinite, (s.ca, s.e_na, s.e_k, s.e_ca, s.e_h, s.e_l, s.c_m, s.tau_ca, s.kd_bk, s.kd_sk, s.dt, s.gain)) &&
        s.ca >= 0.0 &&
        s.c_m > 0.0 &&
        s.tau_ca > 0.0 &&
        s.kd_bk > 0.0 &&
        s.kd_sk > 0.0 &&
        s.dt > 0.0 &&
        s.sub_steps > 0 &&
        s.gain >= 0.0
end

function step!(s::GolgiCellState, I_ext::Float64=0.0; dt::Float64=s.dt)
    if !isfinite(I_ext) || !_valid_state(s)
        return 0
    end

    v_prev = s.v
    v = s.v
    m = s.m
    h = s.h
    p_na = s.p_na
    n = s.n
    a = s.a
    b = s.b
    w = s.w
    m_t = s.m_t
    sg = s.s
    c_n = s.c_n
    r = s.r
    ca = s.ca
    dt_sub = s.dt / s.sub_steps
    input = s.gain * I_ext

    for _ in 1:s.sub_steps
        alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
        beta_m = 4.0 * exp(-(v + 60.0) / 18.0)
        alpha_h = 0.07 * exp(-(v + 58.0) / 20.0)
        beta_h = 1.0 / (1.0 + exp(-(v + 28.0) / 10.0))
        m_next = _gate_alpha_beta(m, alpha_m, beta_m, 5.0, dt_sub)
        h_next = _gate_alpha_beta(h, alpha_h, beta_h, 5.0, dt_sub)
        m_next === nothing && return 0
        h_next === nothing && return 0
        tau_pna = 5.0 + 20.0 / max(0.01, 1.0 + ((v + 48.0) / 10.0)^2)
        p_na_next = _gate_inf(p_na, _boltz(v, -48.0, 5.0), tau_pna, dt_sub)
        p_na_next === nothing && return 0
        alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
        beta_n = 0.125 * exp(-(v + 44.0) / 80.0)
        n_next = _gate_alpha_beta(n, alpha_n, beta_n, 5.0, dt_sub)
        n_next === nothing && return 0
        a_next = _gate_inf(a, _boltz(v, -27.0, 16.0), 2.0, dt_sub)
        b_next = _gate_inf(b, _boltz(v, -80.0, -6.0), 15.0, dt_sub)
        a_next === nothing && return 0
        b_next === nothing && return 0
        tau_w = 100.0 / (3.3 * exp((v + 35.0) / 20.0) + exp(-(v + 35.0) / 20.0))
        w_next = _gate_inf(w, _boltz(v, -35.0, 10.0), tau_w, dt_sub)
        w_next === nothing && return 0
        m_t_next = _gate_inf(m_t, _boltz(v, -52.0, 5.0), 1.0, dt_sub)
        tau_s = 20.0 + 50.0 / max(0.01, 1.0 + ((v + 65.0) / 10.0)^2)
        s_next = _gate_inf(sg, _boltz(v, -60.0, -6.5), tau_s, dt_sub)
        m_t_next === nothing && return 0
        s_next === nothing && return 0
        tau_cn = 2.0 + 10.0 / max(0.01, 1.0 + ((v + 20.0) / 10.0)^2)
        c_n_next = _gate_inf(c_n, _boltz(v, -20.0, 5.0), tau_cn, dt_sub)
        c_n_next === nothing && return 0
        tau_r = 50.0 + 200.0 / max(0.01, 1.0 + ((v + 80.0) / 20.0)^2)
        r_next = _gate_inf(r, _boltz(v, -80.0, -10.0), tau_r, dt_sub)
        r_next === nothing && return 0

        g_cat = s.g_cat * m_t_next^2 * s_next
        g_can = s.g_can * c_n_next^2
        i_ca = g_cat * (v - s.e_ca) + g_can * (v - s.e_ca)
        ca_entry = i_ca < 0.0 ? -i_ca * 0.001 : 0.0
        ca_next = _calcium(ca, ca_entry, s.tau_ca, dt_sub)
        ca_next === nothing && return 0
        ca2 = ca_next^2
        bk_v = _boltz(v, 100.0 - 120.0 * ca2 / (ca2 + s.kd_bk^2), 15.0)
        sk_inf = ca2 / (ca2 + s.kd_sk^2)
        g_na = s.g_na_t * m_next^3 * h_next + s.g_na_p * p_na_next
        g_k = s.g_kdr * n_next^4 + s.g_ka * a_next^3 * b_next + s.g_km * w_next + s.g_bk * bk_v + s.g_sk * sk_inf
        g_ca = g_cat + g_can
        g_h = s.g_h * r_next
        g_total = g_na + g_k + g_ca + g_h + s.g_l
        if !isfinite(g_total) || g_total <= 0.0
            return 0
        end
        steady_v = (input + g_na * s.e_na + g_k * s.e_k + g_ca * s.e_ca + g_h * s.e_h + s.g_l * s.e_l) / g_total
        v_next = steady_v + (v - steady_v) * exp(-(g_total / s.c_m) * dt_sub)
        if !_voltage(v_next) || !isfinite(ca_next) || ca_next < 0.0
            return 0
        end

        v = v_next
        m = m_next
        h = h_next
        p_na = p_na_next
        n = n_next
        a = a_next
        b = b_next
        w = w_next
        m_t = m_t_next
        sg = s_next
        c_n = c_n_next
        r = r_next
        ca = ca_next
    end

    s.v = v
    s.m = m
    s.h = h
    s.p_na = p_na
    s.n = n
    s.a = a
    s.b = b
    s.w = w
    s.m_t = m_t
    s.s = sg
    s.c_n = c_n
    s.r = r
    s.ca = ca
    return (s.v >= 0.0 && v_prev < 0.0) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = GolgiCellState()
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

end # module GolgiCellAccel
