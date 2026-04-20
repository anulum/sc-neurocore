# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for chay_keizer

module ChayKeizerAccel

export step!, simulate, ChayKeizerNeuronState

mutable struct ChayKeizerNeuronState
    v::Float64
    n::Float64
    ca::Float64
    g_ca::Float64
    g_k::Float64
    g_kca::Float64
    g_l::Float64
    e_ca::Float64
    e_k::Float64
    e_l::Float64
    k_d::Float64
    f_ca::Float64
    k_ca::Float64
    dt::Float64
    v_threshold::Float64
end

function ChayKeizerNeuronState()
    ChayKeizerNeuronState(-50.0, 0.01, 0.1, 20.0, 25.0, 12.0, 0.1, 100.0, -75.0, -40.0, 1.0, 0.004, 0.03, 0.02, -20.0)
end

function step!(s::ChayKeizerNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        m_inf = 1.0 / (1.0 + exp(clamp(-(s.v + 25.0) / 8.0, -500.0, 500.0)))
        n_inf = 1.0 / (1.0 + exp(clamp(-(s.v + 18.0) / 14.0, -500.0, 500.0)))
        tau_n = 20.0 / (1.0 + exp(clamp((s.v + 18.0) / 14.0, -500.0, 500.0)))
        q_kca = s.ca / (s.ca + s.k_d)
        i_ca = s.g_ca * m_inf * (s.v - s.e_ca)
        i_k = s.g_k * s.n * (s.v - s.e_k)
        i_kca = s.g_kca * q_kca * (s.v - s.e_k)
        i_l = s.g_l * (s.v - s.e_l)
        s.v += (-i_ca - i_k - i_kca - i_l + I_ext) * s.dt
        s.v = clamp(s.v, -200.0, 200.0)
        s.n += (n_inf - s.n) / max(tau_n, 0.1) * s.dt
        s.n = clamp(s.n, 0.0, 1.0)
        s.ca = max(0.0, s.ca + (-s.f_ca * i_ca - s.k_ca * s.ca) * s.dt)
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ChayKeizerNeuronState()
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

end # module ChayKeizerAccel
