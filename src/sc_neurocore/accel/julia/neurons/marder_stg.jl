# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for marder_stg

module MarderStgAccel

export step!, simulate, MarderSTGNeuronState

mutable struct MarderSTGNeuronState
    v::Float64
    m_na::Float64
    h_na::Float64
    m_cat::Float64
    h_cat::Float64
    m_cas::Float64
    m_a::Float64
    h_a::Float64
    m_kd::Float64
    m_h::Float64
    ca::Float64
    g_na::Float64
    g_cat::Float64
    g_cas::Float64
    g_a::Float64
    g_kca::Float64
    g_kd::Float64
    g_h::Float64
    g_l::Float64
    e_na::Float64
    e_ca::Float64
    e_k::Float64
    e_h::Float64
    e_l::Float64
    ca_decay::Float64
    f_ca::Float64
    dt::Float64
    v_threshold::Float64
end

function MarderSTGNeuronState()
    MarderSTGNeuronState(-60.0, 0.0, 0.9, 0.0, 0.9, 0.0, 0.0, 0.9, 0.0, 0.0, 0.05, 200.0, 2.5, 4.0, 50.0, 25.0, 75.0, 0.01, 0.01, 50.0, 80.0, -80.0, -20.0, -50.0, 0.02, 0.0003, 0.05, -20.0)
end

function _boltz(s::MarderSTGNeuronState, v, v_half, k)
    return 1.0 / (1.0 + exp((v_half - v) / k))
end

function step!(s::MarderSTGNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        m_na_inf = s._boltz(s.v, -25.5, 5.29)
        h_na_inf = s._boltz(s.v, -48.9, -5.18)
        m_cat_inf = s._boltz(s.v, -27.1, 7.2)
        h_cat_inf = s._boltz(s.v, -32.1, -5.5)
        m_cas_inf = s._boltz(s.v, -33.0, 8.1)
        m_a_inf = s._boltz(s.v, -27.2, 8.7)
        h_a_inf = s._boltz(s.v, -56.9, -4.9)
        m_kd_inf = s._boltz(s.v, -12.3, 11.8)
        m_h_inf = s._boltz(s.v, -70.0, -6.0)
        s.m_na = m_na_inf
        s.h_na += (h_na_inf - s.h_na) / 1.5 * s.dt
        s.m_cat += (m_cat_inf - s.m_cat) / 7.2 * s.dt
        s.h_cat += (h_cat_inf - s.h_cat) / 55.0 * s.dt
        s.m_cas += (m_cas_inf - s.m_cas) / 14.0 * s.dt
        s.m_a += (m_a_inf - s.m_a) / 11.6 * s.dt
        s.h_a += (h_a_inf - s.h_a) / 38.6 * s.dt
        s.m_kd += (m_kd_inf - s.m_kd) / 7.2 * s.dt
        s.m_h += (m_h_inf - s.m_h) / 272.0 * s.dt
        kca_act = s.ca / (s.ca + 3.0)
        i_na = s.g_na * s.m_na ^ 3 * s.h_na * (s.v - s.e_na)
        i_cat = s.g_cat * s.m_cat ^ 3 * s.h_cat * (s.v - s.e_ca)
        i_cas = s.g_cas * s.m_cas ^ 3 * (s.v - s.e_ca)
        i_a = s.g_a * s.m_a ^ 3 * s.h_a * (s.v - s.e_k)
        i_kca = s.g_kca * kca_act ^ 4 * (s.v - s.e_k)
        i_kd = s.g_kd * s.m_kd ^ 4 * (s.v - s.e_k)
        i_h = s.g_h * s.m_h * (s.v - s.e_h)
        i_l = s.g_l * (s.v - s.e_l)
        i_total = -i_na - i_cat - i_cas - i_a - i_kca - i_kd - i_h - i_l + I_ext
        s.v += i_total * s.dt
        i_ca_total = i_cat + i_cas
        s.ca = max(0.0, s.ca + (-s.f_ca * i_ca_total - s.ca_decay * s.ca) * s.dt)
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = MarderSTGNeuronState()
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

end # module MarderStgAccel
