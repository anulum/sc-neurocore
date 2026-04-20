# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for dendritic_nmda

module DendriticNmdaAccel

export step!, simulate, DendriticNMDANeuronState

mutable struct DendriticNMDANeuronState
    g_nmda::Float64
    e_nmda::Float64
    mg_conc::Float64
    g_coupling::Float64
    tau_soma::Float64
    tau_dend::Float64
    theta::Float64
    dt::Float64
    v_soma::Float64
    v_dend::Float64
end

function DendriticNMDANeuronState()
    DendriticNMDANeuronState(1.5, 0.0, 1.0, 0.5, 20.0, 50.0, -50.0, 0.1, -65.0, -65.0)
end

function mg_block(s::DendriticNMDANeuronState, v)
    return 1.0 / (1.0 + s.mg_conc / 3.57 * exp(-0.062 * v))
end

function step!(s::DendriticNMDANeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        b = s.mg_block(s.v_dend)
        i_nmda = s.g_nmda * glutamate * b * (s.v_dend - s.e_nmda)
        dv_dend = (-s.v_dend - 65.0 + i_nmda + s.g_coupling * (s.v_soma - s.v_dend)) / s.tau_dend
        s.v_dend += dv_dend * s.dt
        i_dend_to_soma = s.g_coupling * (s.v_dend - s.v_soma)
        dv_soma = (-s.v_soma - 65.0 + i_soma + i_dend_to_soma) / s.tau_soma
        s.v_soma += dv_soma * s.dt
        if s.v_soma >= s.theta
            s.v_soma = -65.0
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = DendriticNMDANeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.g_nmda
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module DendriticNmdaAccel
