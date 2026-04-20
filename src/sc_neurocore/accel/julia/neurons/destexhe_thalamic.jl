# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for destexhe_thalamic

module DestexheThalamicAccel

export step!, simulate, DestexheThalamicNeuronState

mutable struct DestexheThalamicNeuronState
    v::Float64
    h_na::Float64
    n_k::Float64
    m_t::Float64
    h_t::Float64
    g_na::Float64
    g_k::Float64
    g_t::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_ca::Float64
    e_l::Float64
    dt::Float64
    v_threshold::Float64
end

function DestexheThalamicNeuronState()
    DestexheThalamicNeuronState(-65.0, 0.6, 0.3, 0.0, 1.0, 100.0, 10.0, 2.0, 0.05, 50.0, -90.0, 120.0, -70.0, 0.02, -20.0)
end

function step!(s::DestexheThalamicNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        for _ in 1:5
            m_na_inf = 1.0 / (1.0 + exp(-(s.v + 37.0) / 7.0))
            h_na_inf = 1.0 / (1.0 + exp((s.v + 41.0) / 4.0))
            n_k_inf = 1.0 / (1.0 + exp(-(s.v + 25.0) / 12.0))
            m_t_inf = 1.0 / (1.0 + exp(-(s.v + 57.0) / 6.5))
            h_t_inf = 1.0 / (1.0 + exp((s.v + 81.0) / 4.0))
            tau_h_na = 1.0 / (0.128 * exp(-(s.v + 46.0) / 18.0) + 4.0 / (1.0 + exp(-(s.v + 23.0) / 5.0)))
            tau_n_k = (true) ? 1.0 / (0.032 * 5.0 + 0.5 * exp(-(s.v + 40.0) / 40.0)) : 1.0
            tau_h_t = (s.v < -81.0) ? 30.8 + 211.4 * exp((s.v + 115.2) / 5.0) / (1.0 + exp((s.v + 86.0) / 3.2)) : 10.0
            s.h_na += (h_na_inf - s.h_na) / max(tau_h_na, 0.1) * s.dt
            s.n_k += (n_k_inf - s.n_k) / max(tau_n_k, 0.1) * s.dt
            s.m_t = m_t_inf
            s.h_t += (h_t_inf - s.h_t) / max(tau_h_t, 0.1) * s.dt
            i_na = s.g_na * m_na_inf ^ 3 * s.h_na * (s.v - s.e_na)
            i_k = s.g_k * s.n_k ^ 4 * (s.v - s.e_k)
            i_t = s.g_t * s.m_t ^ 2 * s.h_t * (s.v - s.e_ca)
            i_l = s.g_l * (s.v - s.e_l)
            s.v += (-i_na - i_k - i_t - i_l + I_ext) * s.dt
        end
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = DestexheThalamicNeuronState()
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

end # module DestexheThalamicAccel
