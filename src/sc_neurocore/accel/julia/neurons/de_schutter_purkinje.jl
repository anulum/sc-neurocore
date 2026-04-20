# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for de_schutter_purkinje

module DeSchutterPurkinjeAccel

export step!, simulate, DeSchutterPurkinjeNeuronState

mutable struct DeSchutterPurkinjeNeuronState
    v::Float64
    h_na::Float64
    n_k::Float64
    m_cap::Float64
    h_cap::Float64
    q_kca::Float64
    ca::Float64
    g_na::Float64
    g_k::Float64
    g_cap::Float64
    g_kca::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_ca::Float64
    e_l::Float64
    ca_decay::Float64
    f_ca::Float64
    dt::Float64
    v_threshold::Float64
end

function DeSchutterPurkinjeNeuronState()
    DeSchutterPurkinjeNeuronState(-68.0, 0.8, 0.1, 0.0, 0.9, 0.0, 0.0001, 125.0, 10.0, 45.0, 35.0, 0.5, 45.0, -85.0, 135.0, -68.0, 0.02, 0.00024, 0.01, -20.0)
end

function step!(s::DeSchutterPurkinjeNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        for _ in 1:5
            m_na_inf = 1.0 / (1.0 + exp(-(s.v + 35.0) / 7.5))
            h_na_inf = 1.0 / (1.0 + exp((s.v + 55.0) / 7.0))
            n_k_inf = 1.0 / (1.0 + exp(-(s.v + 30.0) / 15.0))
            m_cap_inf = 1.0 / (1.0 + exp(-(s.v + 19.0) / 5.5))
            h_cap_inf = 1.0 / (1.0 + exp((s.v + 48.0) / 7.0))
            q_kca_inf = s.ca / (s.ca + 0.0002)
            tau_h_na = 0.5 + 14.0 / (1.0 + exp((s.v + 40.0) / 12.0))
            tau_n_k = 1.0 + 11.0 / (1.0 + exp((s.v + 15.0) / 8.0))
            tau_m_cap = 0.3
            tau_h_cap = 45.0
            tau_q = 1.0
            s.h_na += (h_na_inf - s.h_na) / tau_h_na * s.dt
            s.n_k += (n_k_inf - s.n_k) / tau_n_k * s.dt
            s.m_cap += (m_cap_inf - s.m_cap) / tau_m_cap * s.dt
            s.h_cap += (h_cap_inf - s.h_cap) / tau_h_cap * s.dt
            s.q_kca += (q_kca_inf - s.q_kca) / tau_q * s.dt
            i_na = s.g_na * m_na_inf ^ 3 * s.h_na * (s.v - s.e_na)
            i_k = s.g_k * s.n_k ^ 4 * (s.v - s.e_k)
            i_cap = s.g_cap * s.m_cap ^ 2 * s.h_cap * (s.v - s.e_ca)
            i_kca = s.g_kca * s.q_kca * (s.v - s.e_k)
            i_l = s.g_l * (s.v - s.e_l)
            s.v += (-i_na - i_k - i_cap - i_kca - i_l + I_ext) * s.dt
            s.ca = max(0.0, s.ca + (-s.f_ca * i_cap - s.ca_decay * s.ca) * s.dt)
        end
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = DeSchutterPurkinjeNeuronState()
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

end # module DeSchutterPurkinjeAccel
