# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for vip_neuron

module VipNeuronAccel

export step!, simulate, VIPNeuronState

mutable struct VIPNeuronState
    v::Float64
    h::Float64
    n::Float64
    a::Float64
    b::Float64
    g_na::Float64
    g_k::Float64
    g_a::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    c_m::Float64
    dt::Float64
    v_threshold::Float64
end

function VIPNeuronState()
    VIPNeuronState(-65.0, 0.8, 0.1, 0.0, 0.9, 35.0, 6.0, 8.0, 0.01, 55.0, -90.0, -65.0, 0.5, 0.025, -20.0)
end

function step!(s::VIPNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        for _ in 1:4
            m_inf = 1.0 / (1.0 + exp(-(s.v + 30.0) / 9.5))
            h_inf = 1.0 / (1.0 + exp((s.v + 53.0) / 7.0))
            tau_h = 0.37 + 2.78 / (1.0 + exp((s.v + 40.5) / 6.0))
            s.h += (h_inf - s.h) / tau_h * s.dt
            n_inf = 1.0 / (1.0 + exp(-(s.v + 30.0) / 10.0))
            tau_n = 0.37 + 1.85 / (1.0 + exp((s.v + 27.0) / 15.0))
            s.n += (n_inf - s.n) / tau_n * s.dt
            a_inf = 1.0 / (1.0 + exp(-(s.v + 50.0) / 20.0))
            b_inf = 1.0 / (1.0 + exp((s.v + 78.0) / 6.0))
            s.a += (a_inf - s.a) / 5.0 * s.dt
            s.b += (b_inf - s.b) / 50.0 * s.dt
            i_na = s.g_na * m_inf ^ 3 * s.h * (s.v - s.e_na)
            i_k = s.g_k * s.n ^ 4 * (s.v - s.e_k)
            i_a = s.g_a * s.a ^ 3 * s.b * (s.v - s.e_k)
            i_l = s.g_l * (s.v - s.e_l)
            s.v += (-i_na - i_k - i_a - i_l + I_ext) / s.c_m * s.dt
        end
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = VIPNeuronState()
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

end # module VipNeuronAccel
