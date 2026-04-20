# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for golgi_cell

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
end

function GolgiCellState()
    GolgiCellState(-60.0, 0.02, 0.85, 0.01, 0.05, 0.1, 0.8, 0.01, 0.01, 0.9, 0.01, 0.1, 0.05, 48.0, 0.2, 16.0, 8.0, 1.0, 0.5, 1.0, 3.0, 1.0, 0.1, 0.05, 55.0, -90.0, 120.0, -40.0, -55.0, 1.0)
end

function step!(s::GolgiCellState, I_ext::Float64=0.0; dt::Float64=0.1)
    inp = s.gain * I_ext
    dt_sub = s.dt / s.sub_steps
    v_prev = s.v
    for _ in 1:s.sub_steps
        v = s.v
        alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
        beta_m = 4.0 * exp(-(v + 60.0) / 18.0)
        alpha_h = 0.07 * exp(-(v + 58.0) / 20.0)
        beta_h = 1.0 / (1.0 + exp(-(v + 28.0) / 10.0))
        s.m += dt_sub * 5.0 * (alpha_m * (1.0 - s.m) - beta_m * s.m)
        s.h += dt_sub * 5.0 * (alpha_h * (1.0 - s.h) - beta_h * s.h)
        pna_inf = _boltz(v, -48.0, 5.0)
        tau_pna = 5.0 + 20.0 / max(0.01, 1.0 + ((v + 48.0) / 10.0) ^ 2)
        s.p_na += dt_sub * (pna_inf - s.p_na) / tau_pna
        alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
        beta_n = 0.125 * exp(-(v + 44.0) / 80.0)
        s.n += dt_sub * 5.0 * (alpha_n * (1.0 - s.n) - beta_n * s.n)
        a_inf = _boltz(v, -27.0, 16.0)
        s.a += dt_sub * (a_inf - s.a) / 2.0
        b_inf = _boltz(v, -80.0, -6.0)
        s.b += dt_sub * (b_inf - s.b) / 15.0
        w_inf = _boltz(v, -35.0, 10.0)
        tau_w = 100.0 / (3.3 * exp((v + 35.0) / 20.0) + exp(-(v + 35.0) / 20.0))
        s.w += dt_sub * (w_inf - s.w) / tau_w
        mt_inf = _boltz(v, -52.0, 5.0)
        s.m_t += dt_sub * (mt_inf - s.m_t) / 1.0
        s_inf = _boltz(v, -60.0, -6.5)
        tau_s = 20.0 + 50.0 / max(0.01, 1.0 + ((v + 65.0) / 10.0) ^ 2)
        s.s += dt_sub * (s_inf - s.s) / tau_s
        cn_inf = _boltz(v, -20.0, 5.0)
        tau_cn = 2.0 + 10.0 / max(0.01, 1.0 + ((v + 20.0) / 10.0) ^ 2)
        s.c_n += dt_sub * (cn_inf - s.c_n) / tau_cn
        r_inf = _boltz(v, -80.0, -10.0)
        tau_r = 50.0 + 200.0 / max(0.01, 1.0 + ((v + 80.0) / 20.0) ^ 2)
        s.r += dt_sub * (r_inf - s.r) / tau_r
        for attr in ('m", "h", "p_na", "n", "a", "b", "w", "m_t", "s", "c_n", "r')
            setattr(self, attr, max(0.0, min(1.0, getattr(self, attr))))
        end
        i_cat = s.g_cat * s.m_t ^ 2 * s.s * (v - s.e_ca)
        i_can = s.g_can * s.c_n ^ 2 * (v - s.e_ca)
        ca_entry = (i_cat + i_can < 0.0) ? -(i_cat + i_can) * 0.001 : 0.0
        s.ca += dt_sub * (ca_entry - s.ca / s.tau_ca)
        s.ca = max(0.0, s.ca)
        ca2 = s.ca ^ 2
        kd2 = s.kd_bk ^ 2
        bk_v = _boltz(v, 100.0 - 120.0 * ca2 / (ca2 + kd2), 15.0)
        sk_inf = ca2 / (ca2 + s.kd_sk ^ 2)
        i_na_t = s.g_na_t * s.m ^ 3 * s.h * (v - s.e_na)
        i_na_p = s.g_na_p * s.p_na * (v - s.e_na)
        i_kdr = s.g_kdr * s.n ^ 4 * (v - s.e_k)
        i_ka = s.g_ka * s.a ^ 3 * s.b * (v - s.e_k)
        i_km = s.g_km * s.w * (v - s.e_k)
        i_bk = s.g_bk * bk_v * (v - s.e_k)
        i_sk = s.g_sk * sk_inf * (v - s.e_k)
        i_h = s.g_h * s.r * (v - s.e_h)
        i_l = s.g_l * (v - s.e_l)
        dv_val = (-(i_na_t + i_na_p + i_kdr + i_ka + i_km + i_cat + i_can + i_bk + i_sk + i_h + i_l) + inp) / s.c_m
        s.v += dt_sub * dv_val
    end
    s.v = max(-100.0, min(60.0, s.v))
    if ! isfinite(s.v)
        s.v = -60.0
    end
    if ! isfinite(s.ca)
        s.ca = 0.05
    end
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
