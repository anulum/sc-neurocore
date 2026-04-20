# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for granule_cell

module GranuleCellAccel

export step!, simulate, GranuleCellState

mutable struct GranuleCellState
    v::Float64
    m::Float64
    h::Float64
    n::Float64
    a::Float64
    b::Float64
    m_t::Float64
    s::Float64
    ca::Float64
    r::Float64
    c_m::Float64
    g_na::Float64
    g_kdr::Float64
    g_ka::Float64
    g_t::Float64
    g_kca::Float64
    g_h::Float64
    g_l::Float64
    g_tonic::Float64
    e_na::Float64
    e_k::Float64
    e_ca::Float64
    e_h::Float64
    e_l::Float64
    e_gaba::Float64
    tau_ca::Float64
    kd_kca::Float64
    dt::Float64
    sub_steps::Float64
    gain::Float64
end

function GranuleCellState()
    GranuleCellState(-70.0, 0.02, 0.85, 0.05, 0.1, 0.8, 0.01, 0.95, 0.05, 0.1, 1.0, 17.0, 9.0, 1.0, 0.5, 3.5, 0.03, 0.1, 0.2, 87.4, -84.7, 129.3, -40.0, -58.0, -75.0, 10.0, 0.2, 0.5, 4.0, 1.0)
end

function _boltz(s::GranuleCellState, v, vh, k)
    return 1.0 / (1.0 + exp(-(v - vh) / k))
end

function step!(s::GranuleCellState, I_ext::Float64=0.0; dt::Float64=0.1)
    inp = s.gain * I_ext
    dt_sub = s.dt / s.sub_steps
    v_prev = s.v
    for _ in 1:s.sub_steps
        v = s.v
        bz = s._boltz
        m_inf = bz(v, -30.0, 7.0)
        tau_m = 0.1 + 0.3 / max(0.01, 1.0 + ((v + 30.0) / 10.0) ^ 2)
        s.m += dt_sub * (m_inf - s.m) / tau_m
        h_inf = bz(v, -52.0, -6.0)
        tau_h = 0.5 + 5.0 / max(0.01, 1.0 + ((v + 50.0) / 15.0) ^ 2)
        s.h += dt_sub * (h_inf - s.h) / tau_h
        n_inf = bz(v, -35.0, 8.0)
        tau_n = 1.0 + 5.0 / max(0.01, 1.0 + ((v + 35.0) / 15.0) ^ 2)
        s.n += dt_sub * (n_inf - s.n) / tau_n
        a_inf = bz(v, -50.0, 20.0)
        s.a += dt_sub * (a_inf - s.a) / 2.0
        b_inf = bz(v, -70.0, -6.0)
        s.b += dt_sub * (b_inf - s.b) / 50.0
        mt_inf = bz(v, -52.0, 5.0)
        s.m_t += dt_sub * (mt_inf - s.m_t) / 1.0
        s_inf = bz(v, -60.0, -6.5)
        tau_s = 20.0 + 50.0 / max(0.01, 1.0 + ((v + 65.0) / 10.0) ^ 2)
        s.s += dt_sub * (s_inf - s.s) / tau_s
        r_inf = bz(v, -80.0, -10.0)
        tau_r = 50.0 + 200.0 / max(0.01, 1.0 + ((v + 80.0) / 20.0) ^ 2)
        s.r += dt_sub * (r_inf - s.r) / tau_r
        for attr in ('m", "h", "n", "a", "b", "m_t", "s", "r')
            setattr(self, attr, max(0.0, min(1.0, getattr(self, attr))))
        end
        i_ca_t = s.g_t * s.m_t ^ 2 * s.s * (v - s.e_ca)
        ca_entry = (i_ca_t < 0.0) ? -i_ca_t * 0.001 : 0.0
        s.ca += dt_sub * (-s.ca / s.tau_ca + ca_entry)
        s.ca = max(0.0, s.ca)
        kca_inf = s.ca ^ 2 / (s.ca ^ 2 + s.kd_kca ^ 2)
        i_na = s.g_na * s.m ^ 3 * s.h * (v - s.e_na)
        i_kdr = s.g_kdr * s.n ^ 4 * (v - s.e_k)
        i_ka = s.g_ka * s.a ^ 3 * s.b * (v - s.e_k)
        i_kca = s.g_kca * kca_inf * (v - s.e_k)
        i_h = s.g_h * s.r * (v - s.e_h)
        i_l = s.g_l * (v - s.e_l)
        i_gaba = s.g_tonic * (v - s.e_gaba)
        dv_val = (-(i_na + i_kdr + i_ka + i_ca_t + i_kca + i_h + i_l + i_gaba) + inp) / s.c_m
        s.v += dt_sub * dv_val
    end
    s.v = max(-100.0, min(60.0, s.v))
    if ! isfinite(s.v)
        s.v = -70.0
    end
    return (s.v >= 0.0 && v_prev < 0.0) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = GranuleCellState()
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

end # module GranuleCellAccel
