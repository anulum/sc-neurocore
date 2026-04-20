# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for bertram_phantom

module BertramPhantomAccel

export step!, simulate, BertramPhantomBursterState

mutable struct BertramPhantomBursterState
    v::Float64
    s1::Float64
    s2::Float64
    g_ca::Float64
    g_k::Float64
    g_s1::Float64
    g_s2::Float64
    g_l::Float64
    e_ca::Float64
    e_k::Float64
    e_l::Float64
    c_m::Float64
    v_m::Float64
    s_m::Float64
    v_n::Float64
    s_n::Float64
    v_s1::Float64
    s_s1::Float64
    v_s2::Float64
    s_s2::Float64
    tau_s1::Float64
    tau_s2::Float64
    dt::Float64
    v_threshold::Float64
end

function BertramPhantomBursterState()
    BertramPhantomBursterState(-50.0, 0.1, 0.1, 3.6, 10.0, 4.0, 4.0, 0.2, 25.0, -75.0, -40.0, 5.3, -20.0, 12.0, -16.0, 5.6, -40.0, 10.0, -42.0, 0.4, 20000.0, 100000.0, 0.5, -20.0)
end

function _boltz(s::BertramPhantomBursterState, v, vh, k)
    return 1.0 / (1.0 + exp((vh - v) / k))
end

function step!(s::BertramPhantomBursterState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        m_inf = s._boltz(s.v, s.v_m, s.s_m)
        n_inf = s._boltz(s.v, s.v_n, s.s_n)
        s1_inf = s._boltz(s.v, s.v_s1, s.s_s1)
        s2_inf = s._boltz(s.v, s.v_s2, s.s_s2)
        i_ca = s.g_ca * m_inf * (s.v - s.e_ca)
        i_k = s.g_k * n_inf * (s.v - s.e_k)
        i_s1 = s.g_s1 * s.s1 * (s.v - s.e_k)
        i_s2 = s.g_s2 * s.s2 * (s.v - s.e_k)
        i_l = s.g_l * (s.v - s.e_l)
        s.v += (-i_ca - i_k - i_s1 - i_s2 - i_l + I_ext) / s.c_m * s.dt
        s.s1 += (s1_inf - s.s1) / s.tau_s1 * s.dt
        s.s2 += (s2_inf - s.s2) / s.tau_s2 * s.dt
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = BertramPhantomBursterState()
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

end # module BertramPhantomAccel
