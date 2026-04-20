# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for hill_tononi

module HillTononiAccel

export step!, simulate, HillTononiNeuronState

mutable struct HillTononiNeuronState
    v::Float64
    h_na::Float64
    n_k::Float64
    m_h::Float64
    h_t::Float64
    na_i::Float64
    g_na::Float64
    g_k::Float64
    g_h::Float64
    g_t::Float64
    g_kna::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_h::Float64
    e_ca::Float64
    e_l::Float64
    na_pump_max::Float64
    na_eq::Float64
    dt::Float64
    v_threshold::Float64
end

function HillTononiNeuronState()
    HillTononiNeuronState(-65.0, 0.6, 0.3, 0.0, 0.9, 5.0, 50.0, 5.0, 1.0, 3.0, 1.33, 0.02, 50.0, -90.0, -43.0, 120.0, -70.0, 20.0, 9.5, 0.05, -20.0)
end

function step!(s::HillTononiNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        m_na_inf = 1.0 / (1.0 + exp(-(s.v + 38.0) / 10.0))
        h_na_inf = 1.0 / (1.0 + exp((s.v + 43.0) / 6.0))
        n_k_inf = 1.0 / (1.0 + exp(-(s.v + 27.0) / 11.5))
        m_h_inf = 1.0 / (1.0 + exp((s.v + 75.0) / 5.5))
        m_t_inf = 1.0 / (1.0 + exp(-(s.v + 59.0) / 6.2))
        h_t_inf = 1.0 / (1.0 + exp((s.v + 83.0) / 4.0))
        w_kna = 0.37 / (1.0 + (38.7 / max(s.na_i, 0.01)) ^ 3.5)
        tau_h_na = 1.0 + 10.0 / (1.0 + exp((s.v + 40.0) / 10.0))
        tau_n_k = 5.0 + 47.0 * exp(-((s.v + 50.0) / 25.0) ^ 2)
        tau_m_h = 20.0 + 1000.0 / (exp((s.v + 71.5) / 14.2) + exp(-(s.v + 89.0) / 11.6))
        tau_h_t = (s.v < -81.0) ? 30.8 + 211.4 * exp((s.v + 115.2) / 5.0) / (1.0 + exp((s.v + 86.0) / 3.2)) : 10.0
        s.h_na += (h_na_inf - s.h_na) / max(tau_h_na, 0.1) * s.dt
        s.n_k += (n_k_inf - s.n_k) / max(tau_n_k, 0.1) * s.dt
        s.m_h += (m_h_inf - s.m_h) / max(tau_m_h, 1.0) * s.dt
        s.h_t += (h_t_inf - s.h_t) / max(tau_h_t, 0.1) * s.dt
        i_na = s.g_na * m_na_inf ^ 3 * s.h_na * (s.v - s.e_na)
        i_k = s.g_k * s.n_k ^ 4 * (s.v - s.e_k)
        i_h = s.g_h * s.m_h * (s.v - s.e_h)
        i_t = s.g_t * m_t_inf ^ 2 * s.h_t * (s.v - s.e_ca)
        i_kna = s.g_kna * w_kna * (s.v - s.e_k)
        i_l = s.g_l * (s.v - s.e_l)
        s.v += (-i_na - i_k - i_h - i_t - i_kna - i_l + I_ext) * s.dt
        s.na_i += (-0.001 * i_na - s.na_pump_max * (s.na_i / (s.na_i + s.na_eq))) * s.dt
        s.na_i = max(s.na_i, 0.0)
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = HillTononiNeuronState()
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

end # module HillTononiAccel
