# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for pinsky_rinzel

module PinskyRinzelAccel

export step!, simulate, PinskyRinzelNeuronState

mutable struct PinskyRinzelNeuronState
    v_s::Float64
    v_d::Float64
    h::Float64
    n::Float64
    s::Float64
    c::Float64
    q::Float64
    gc::Float64
    p::Float64
    g_na::Float64
    g_kdr::Float64
    g_ca::Float64
    g_kahp::Float64
    g_kc::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_ca::Float64
    e_l::Float64
    dt::Float64
    v_threshold::Float64
end

function PinskyRinzelNeuronState()
    PinskyRinzelNeuronState(-60.0, -60.0, 0.9, 0.1, 0.0, 0.0, 0.0, 2.1, 0.5, 30.0, 15.0, 10.0, 0.8, 15.0, 0.1, 60.0, -75.0, 80.0, -60.0, 0.02, -20.0)
end

function step!(s::PinskyRinzelNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v_s
        am = (abs(s.v_s + 54.0) > 1e-06) ? 0.32 * (s.v_s + 54.0) / (1.0 - exp(-(s.v_s + 54.0) / 4.0)) : 8.0
        bm = (abs(s.v_s + 27.0) > 1e-06) ? 0.28 * (s.v_s + 27.0) / (exp((s.v_s + 27.0) / 5.0) - 1.0) : 5.6
        m_inf = am / (am + bm)
        ah = 0.128 * exp(-(s.v_s + 50.0) / 18.0)
        bh = 4.0 / (1.0 + exp(-(s.v_s + 27.0) / 5.0))
        an = (abs(s.v_s + 52.0) > 1e-06) ? 0.032 * (s.v_s + 52.0) / (1.0 - exp(-(s.v_s + 52.0) / 5.0)) : 0.32
        bn = 0.5 * exp(-(s.v_s + 57.0) / 40.0)
        s_inf = 1.0 / (1.0 + exp(-(s.v_d + 20.0) / 9.0))
        c_inf = (s.c > 0) ? min(s.c, 1.0) : 0.0
        i_na = s.g_na * m_inf ^ 2 * s.h * (s.v_s - s.e_na)
        i_kdr = s.g_kdr * s.n * (s.v_s - s.e_k)
        i_ls = s.g_l * (s.v_s - s.e_l)
        i_ds = s.gc / s.p * (s.v_s - s.v_d)
        i_ca = s.g_ca * s.s ^ 2 * (s.v_d - s.e_ca)
        i_kahp = s.g_kahp * s.q * (s.v_d - s.e_k)
        chi = (s.v_d <= 50.0) ? min(s.v_d / 250.0 + 0.5, 1.0) : 2.0
        i_kc = s.g_kc * s.c * chi * (s.v_d - s.e_k)
        i_ld = s.g_l * (s.v_d - s.e_l)
        i_sd = s.gc / (1 - s.p) * (s.v_d - s.v_s)
        s.v_s += (-i_na - i_kdr - i_ls - i_ds + current_soma / s.p) * s.dt
        s.v_d += (-i_ca - i_kahp - i_kc - i_ld - i_sd + current_dend / (1 - s.p)) * s.dt
        s.h += (ah * (1 - s.h) - bh * s.h) * s.dt
        s.n += (an * (1 - s.n) - bn * s.n) * s.dt
        s.s += (s_inf - s.s) / 5.0 * s.dt
        s.c = max(0.0, s.c + (-0.13 * i_ca - 0.075 * s.c) * s.dt)
        q_inf = min(s.c / (s.c + 2.0), 1.0)
        s.q += (q_inf - s.q) / 100.0 * s.dt
        return (s.v_s >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = PinskyRinzelNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v_s
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module PinskyRinzelAccel
