# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for av_ron_cardiac

module AvRonCardiacAccel

export step!, simulate, AvRonCardiacNeuronState

mutable struct AvRonCardiacNeuronState
    v::Float64
    h::Float64
    n::Float64
    s::Float64
    g_na::Float64
    g_k::Float64
    g_s::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_s::Float64
    e_l::Float64
    dt::Float64
    v_threshold::Float64
end

function AvRonCardiacNeuronState()
    AvRonCardiacNeuronState(-60.0, 0.6, 0.3, 0.5, 80.0, 40.0, 20.0, 0.1, 40.0, -80.0, -25.0, -60.0, 0.02, -20.0)
end

function step!(s::AvRonCardiacNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        m_inf = 1.0 / (1.0 + exp(-(s.v + 40.0) / 7.0))
        h_inf = 1.0 / (1.0 + exp((s.v + 45.0) / 5.0))
        n_inf = 1.0 / (1.0 + exp(-(s.v + 40.0) / 15.0))
        s_inf = 1.0 / (1.0 + exp((s.v + 35.0) / 3.0))
        tau_h = 1.0 + 12.0 / (1.0 + exp((s.v + 50.0) / 8.0))
        tau_n = 1.0 + 8.0 / (1.0 + exp((s.v + 35.0) / 8.0))
        tau_s = 200.0 + 1000.0 / (1.0 + exp((s.v + 30.0) / 5.0))
        s.h += (h_inf - s.h) / tau_h * s.dt
        s.n += (n_inf - s.n) / tau_n * s.dt
        s.s += (s_inf - s.s) / tau_s * s.dt
        i_na = s.g_na * m_inf ^ 3 * s.h * (s.v - s.e_na)
        i_k = s.g_k * s.n ^ 4 * (s.v - s.e_k)
        i_s = s.g_s * s.s * (s.v - s.e_s)
        i_l = s.g_l * (s.v - s.e_l)
        s.v += (-i_na - i_k - i_s - i_l + I_ext) * s.dt
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = AvRonCardiacNeuronState()
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

end # module AvRonCardiacAccel
