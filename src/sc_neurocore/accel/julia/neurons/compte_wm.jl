# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for compte_wm

module CompteWmAccel

export step!, simulate, CompteWMNeuronState

mutable struct CompteWMNeuronState
    v::Float64
    s_ampa::Float64
    s_nmda::Float64
    x_nmda::Float64
    s_gaba::Float64
    g_l::Float64
    g_ampa::Float64
    g_nmda::Float64
    g_gaba::Float64
    e_l::Float64
    e_exc::Float64
    e_inh::Float64
    c_m::Float64
    mg::Float64
    tau_ampa::Float64
    tau_nmda::Float64
    tau_x::Float64
    alpha_nmda::Float64
    v_threshold::Float64
    v_reset::Float64
    dt::Float64
end

function CompteWMNeuronState()
    CompteWMNeuronState(-70.0, 0.0, 0.0, 0.0, 0.0, 0.025, 0.005, 0.165, 0.013, -70.0, 0.0, -70.0, 0.5, 1.0, 2.0, 100.0, 2.0, 0.5, -50.0, -55.0, 0.1)
end

function _mg_block(s::CompteWMNeuronState, v)
    return 1.0 / (1.0 + s.mg / 3.57 * exp(-0.062 * v))
end

function step!(s::CompteWMNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        if spike_in
            s.s_ampa += 1.0
            s.x_nmda += 1.0
        end
        s.s_ampa *= exp(-s.dt / s.tau_ampa)
        s.s_nmda += (-s.s_nmda / s.tau_nmda + s.alpha_nmda * s.x_nmda * (1.0 - s.s_nmda)) * s.dt
        s.x_nmda *= exp(-s.dt / s.tau_x)
        s.s_gaba *= exp(-s.dt / 5.0)
        b = s._mg_block(s.v)
        i_l = s.g_l * (s.v - s.e_l)
        i_ampa = s.g_ampa * s.s_ampa * (s.v - s.e_exc)
        i_nmda = s.g_nmda * b * s.s_nmda * (s.v - s.e_exc)
        i_gaba = s.g_gaba * s.s_gaba * (s.v - s.e_inh)
        s.v += (-i_l - i_ampa - i_nmda - i_gaba + I_ext) / s.c_m * s.dt
        if s.v >= s.v_threshold
            s.v = s.v_reset
            s.s_gaba += 1.0
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = CompteWMNeuronState()
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

end # module CompteWmAccel
