# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for hay_l5

module HayL5Accel

export step!, simulate, HayL5PyramidalNeuronState

mutable struct HayL5PyramidalNeuronState
    v_s::Float64
    h_na::Float64
    n_k::Float64
    v_t::Float64
    m_ca::Float64
    h_ca::Float64
    m_ih::Float64
    v_a::Float64
    ca_a::Float64
    g_na::Float64
    g_k::Float64
    g_l_s::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    g_ca_t::Float64
    g_ih::Float64
    g_l_t::Float64
    e_ca::Float64
    e_ih::Float64
    g_ca_a::Float64
    g_kca::Float64
    g_l_a::Float64
    g_st::Float64
    g_ta::Float64
    p_s::Float64
    p_t::Float64
    p_a::Float64
    ca_decay::Float64
    f_ca::Float64
end

function HayL5PyramidalNeuronState()
    HayL5PyramidalNeuronState(-75.0, 0.9, 0.1, -75.0, 0.0, 1.0, 0.0, -75.0, 0.0001, 300.0, 40.0, 0.03, 50.0, -85.0, -75.0, 2.0, 0.02, 0.03, 140.0, -45.0, 1.5, 2.5, 0.03, 1.5, 0.8, 0.15, 0.25, 0.6, 200.0, 0.0002)
end

function step!(s::HayL5PyramidalNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_s_prev = s.v_s
        for _ in 1:4
            m_na_inf = 1.0 / (1.0 + exp(-(s.v_s + 38.0) / 7.0))
            h_na_inf = 1.0 / (1.0 + exp((s.v_s + 65.0) / 6.0))
            n_k_inf = 1.0 / (1.0 + exp(-(s.v_s + 25.0) / 12.0))
            tau_h = 0.5 + 14.0 / (1.0 + exp((s.v_s + 35.0) / 10.0))
            tau_n = 1.0 + 5.0 / (1.0 + exp((s.v_s + 30.0) / 10.0))
            s.h_na += (h_na_inf - s.h_na) / tau_h * s.dt
            s.n_k += (n_k_inf - s.n_k) / tau_n * s.dt
            i_na = s.g_na * m_na_inf ^ 3 * s.h_na * (s.v_s - s.e_na)
            i_k = s.g_k * s.n_k ^ 4 * (s.v_s - s.e_k)
            i_l_s = s.g_l_s * (s.v_s - s.e_l)
            i_st = s.g_st * (s.v_s - s.v_t) / s.p_s
            m_ca_inf = 1.0 / (1.0 + exp(-(s.v_t + 27.0) / 7.0))
            h_ca_inf = 1.0 / (1.0 + exp((s.v_t + 52.0) / 5.0))
            m_ih_inf = 1.0 / (1.0 + exp((s.v_t + 75.0) / 5.5))
            tau_m_ca = 1.0
            tau_h_ca = 20.0
            tau_ih = 50.0
            s.m_ca += (m_ca_inf - s.m_ca) / tau_m_ca * s.dt
            s.h_ca += (h_ca_inf - s.h_ca) / tau_h_ca * s.dt
            s.m_ih += (m_ih_inf - s.m_ih) / tau_ih * s.dt
            i_ca_t = s.g_ca_t * s.m_ca ^ 2 * s.h_ca * (s.v_t - s.e_ca)
            i_ih = s.g_ih * s.m_ih * (s.v_t - s.e_ih)
            i_l_t = s.g_l_t * (s.v_t - s.e_l)
            i_ts = s.g_st * (s.v_t - s.v_s) / s.p_t
            i_ta = s.g_ta * (s.v_t - s.v_a) / s.p_t
            m_ca_a_inf = 1.0 / (1.0 + exp(-(s.v_a + 30.0) / 5.0))
            kca_act = s.ca_a / (s.ca_a + 0.001)
            i_ca_a = s.g_ca_a * m_ca_a_inf ^ 2 * (s.v_a - s.e_ca)
            i_kca = s.g_kca * kca_act * (s.v_a - s.e_k)
            i_l_a = s.g_l_a * (s.v_a - s.e_l)
            i_at = s.g_ta * (s.v_a - s.v_t) / s.p_a
            s.v_s += (-i_na - i_k - i_l_s - i_st + current_soma / s.p_s) / s.c_m * s.dt
            s.v_t += (-i_ca_t - i_ih - i_l_t - i_ts - i_ta) / s.c_m * s.dt
            s.v_a += (-i_ca_a - i_kca - i_l_a - i_at + current_tuft / s.p_a) / s.c_m * s.dt
            s.ca_a = max(0.0, s.ca_a + (-s.f_ca * i_ca_a - s.ca_a / s.ca_decay) * s.dt)
        end
        return (s.v_s >= s.v_threshold && v_s_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = HayL5PyramidalNeuronState()
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

end # module HayL5Accel
