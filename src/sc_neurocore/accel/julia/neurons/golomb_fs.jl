# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for golomb_fs

module GolombFsAccel

export step!, simulate, GolombFSNeuronState

mutable struct GolombFSNeuronState
    v::Float64
    h::Float64
    n::Float64
    p::Float64
    g_na::Float64
    g_kd::Float64
    g_kv3::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    c_m::Float64
    dt::Float64
    v_threshold::Float64
end

function GolombFSNeuronState()
    GolombFSNeuronState(-65.0, 0.9, 0.1, 0.0, 112.5, 225.0, 150.0, 0.25, 50.0, -90.0, -70.0, 1.0, 0.01, -20.0)
end

function step!(s::GolombFSNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        for _ in 1:10
            m_inf = 1.0 / (1.0 + exp(-(s.v + 24.0) / 11.5))
            h_inf = 1.0 / (1.0 + exp((s.v + 58.3) / 6.7))
            tau_h = 0.5 + 14.0 / (1.0 + exp((s.v + 60.0) / 12.0))
            n_inf = 1.0 / (1.0 + exp(-(s.v + 12.4) / 6.8))
            tau_n = 0.087 + 11.4 / (1.0 + exp((s.v + 14.6) / 8.6))
            p_inf = 1.0 / (1.0 + exp(-(s.v + 3.0) / 8.0))
            tau_p = 0.1 + 4.0 / (1.0 + exp((s.v + 25.0) / 10.0))
            s.h += (h_inf - s.h) / tau_h * s.dt
            s.n += (n_inf - s.n) / tau_n * s.dt
            s.p += (p_inf - s.p) / tau_p * s.dt
            i_na = s.g_na * m_inf ^ 3 * s.h * (s.v - s.e_na)
            i_kd = s.g_kd * s.n ^ 4 * (s.v - s.e_k)
            i_kv3 = s.g_kv3 * s.p ^ 2 * (s.v - s.e_k)
            i_l = s.g_l * (s.v - s.e_l)
            s.v += (-i_na - i_kd - i_kv3 - i_l + I_ext) / s.c_m * s.dt
        end
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = GolombFSNeuronState()
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

end # module GolombFsAccel
