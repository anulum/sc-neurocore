# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for gutkin_ermentrout

module GutkinErmentroutAccel

export step!, simulate, GutkinErmentroutNeuronState

mutable struct GutkinErmentroutNeuronState
    v::Float64
    n::Float64
    g_na::Float64
    g_k::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    dt::Float64
    v_threshold::Float64
end

function GutkinErmentroutNeuronState()
    GutkinErmentroutNeuronState(-65.0, 0.1, 20.0, 10.0, 8.0, 60.0, -90.0, -80.0, 0.05, -20.0)
end

function step!(s::GutkinErmentroutNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        m_inf = 1.0 / (1.0 + exp(-(s.v + 20.0) / 15.0))
        n_inf = 1.0 / (1.0 + exp(-(s.v + 25.0) / 5.0))
        tau_n = 1.0
        s.n += (n_inf - s.n) / tau_n * s.dt
        i_na = s.g_na * m_inf * (s.v - s.e_na)
        i_k = s.g_k * s.n * (s.v - s.e_k)
        i_l = s.g_l * (s.v - s.e_l)
        s.v += (-i_na - i_k - i_l + I_ext) * s.dt
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = GutkinErmentroutNeuronState()
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

end # module GutkinErmentroutAccel
