# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for booth_rinzel

module BoothRinzelAccel

export step!, simulate, BoothRinzelNeuronState

mutable struct BoothRinzelNeuronState
    vs::Float64
    vd::Float64
    h::Float64
    n::Float64
    q::Float64
    ca::Float64
    p::Float64
    gc::Float64
    g_na::Float64
    g_k::Float64
    g_ca::Float64
    g_kca::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_ca::Float64
    e_l::Float64
    c_m::Float64
    alpha_ca::Float64
    k_ca::Float64
    f_ca::Float64
    dt::Float64
    v_threshold::Float64
end

function BoothRinzelNeuronState()
    BoothRinzelNeuronState(-65.0, -65.0, 0.9, 0.0, 0.0, 0.0, 0.5, 0.1, 120.0, 20.0, 14.0, 5.0, 0.51, 55.0, -80.0, 80.0, -60.0, 1.0, 0.009, 0.18, 0.0025, 0.025, -20.0)
end

function _safe_exp(s::BoothRinzelNeuronState, x)
    return Float64(exp(clamp(x, -500, 500)))
end

function step!(s::BoothRinzelNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        vs_prev = s.vs
        for _ in 1:4
            m_inf = 1.0 / (1.0 + s._safe_exp(-(s.vs + 35.0) / 7.8))
            h_inf = 1.0 / (1.0 + s._safe_exp((s.vs + 55.0) / 7.0))
            tau_h = 30.0 / (s._safe_exp((s.vs + 50.0) / 15.0) + s._safe_exp(-(s.vs + 50.0) / 16.0) + 1e-12)
            n_inf = 1.0 / (1.0 + s._safe_exp(-(s.vs + 28.0) / 15.0))
            tau_n = 7.0 / (s._safe_exp((s.vs + 40.0) / 40.0) + s._safe_exp(-(s.vs + 40.0) / 50.0) + 1e-12)
            s.h += (h_inf - s.h) / tau_h * s.dt
            s.h = Float64(clamp(s.h, 0, 1))
            s.n += (n_inf - s.n) / tau_n * s.dt
            s.n = Float64(clamp(s.n, 0, 1))
            i_na = s.g_na * m_inf ^ 3 * s.h * (s.vs - s.e_na)
            i_k = s.g_k * s.n ^ 4 * (s.vs - s.e_k)
            i_ls = s.g_l * (s.vs - s.e_l)
            i_coup_s = s.gc * (s.vs - s.vd) / s.p
            dvs = (-i_na - i_k - i_ls - i_coup_s + I_ext / s.p) / s.c_m * s.dt
            s_inf = 1.0 / (1.0 + s._safe_exp(-(s.vd + 22.0) / 5.0))
            q_inf = 1.0 / (1.0 + s._safe_exp(-(s.vd + 35.0) / 2.0))
            tau_q = 400.0
            s.q += (q_inf - s.q) / tau_q * s.dt
            s.q = Float64(clamp(s.q, 0, 1))
            i_ca = s.g_ca * s_inf ^ 2 * (s.vd - s.e_ca)
            chi = min(s.ca / 250.0, 1.0)
            i_kca = s.g_kca * chi * (s.vd - s.e_k)
            i_ld = s.g_l * (s.vd - s.e_l)
            i_coup_d = s.gc * (s.vd - s.vs) / (1.0 - s.p)
            dvd = (-i_ca - i_kca - i_ld - i_coup_d) / s.c_m * s.dt
            s.ca += s.f_ca * (-s.alpha_ca * i_ca - s.k_ca * s.ca) * s.dt
            s.ca = max(s.ca, 0.0)
            s.vs = Float64(clamp(s.vs + dvs, -200, 100))
            s.vd = Float64(clamp(s.vd + dvd, -200, 100))
        end
        return (s.vs >= s.v_threshold && vs_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = BoothRinzelNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.vs
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module BoothRinzelAccel
