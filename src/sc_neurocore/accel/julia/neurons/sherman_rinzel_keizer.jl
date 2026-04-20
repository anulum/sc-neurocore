# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for sherman_rinzel_keizer

module ShermanRinzelKeizerAccel

export step!, simulate, ShermanRinzelKeizerNeuronState

mutable struct ShermanRinzelKeizerNeuronState
    v::Float64
    n::Float64
    s::Float64
    g_ca::Float64
    g_k::Float64
    g_s::Float64
    e_ca::Float64
    e_k::Float64
    tau_s::Float64
    dt::Float64
    v_threshold::Float64
end

function ShermanRinzelKeizerNeuronState()
    ShermanRinzelKeizerNeuronState(-50.0, 0.1, 0.1, 3.6, 10.0, 4.0, 25.0, -75.0, 5000.0, 0.5, -20.0)
end

function step!(s::ShermanRinzelKeizerNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        m_inf = 1.0 / (1.0 + exp(-(s.v + 20.0) / 12.0))
        n_inf = 1.0 / (1.0 + exp(-(s.v + 16.0) / 5.0))
        s_inf = 1.0 / (1.0 + exp(-(s.v + 35.0) / 10.0))
        tau_n = 9.09
        i_ca = s.g_ca * m_inf * (s.v - s.e_ca)
        i_k = s.g_k * s.n * (s.v - s.e_k)
        i_s = s.g_s * s.s * (s.v - s.e_k)
        s.v += (-i_ca - i_k - i_s + I_ext) * s.dt
        s.n += (n_inf - s.n) / tau_n * s.dt
        s.s += (s_inf - s.s) / s.tau_s * s.dt
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ShermanRinzelKeizerNeuronState()
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

end # module ShermanRinzelKeizerAccel
