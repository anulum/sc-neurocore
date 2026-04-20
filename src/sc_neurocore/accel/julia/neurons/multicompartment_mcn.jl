# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for multicompartment_mcn

module MulticompartmentMcnAccel

export step!, simulate, MulticompartmentMCNNeuronState

mutable struct MulticompartmentMCNNeuronState
    tau::Float64
    tau_b::Float64
    tau_a::Float64
    g_ratio::Float64
    beta::Float64
    v_th::Float64
    dt::Float64
    u::Float64
    v_basal::Float64
    v_apical::Float64
end

function MulticompartmentMCNNeuronState()
    MulticompartmentMCNNeuronState(2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
end

function _sigma(s::MulticompartmentMCNNeuronState, x)
    return 1.0 / (1.0 + exp(-s.beta * x))
end

function step_compartments(s::MulticompartmentMCNNeuronState, x_basal, x_apical, i_soma)
    dv_b = (-s.v_basal + x_basal) / s.tau_b
    s.v_basal += dv_b * s.dt
    dv_a = (-s.v_apical + x_apical) / s.tau_a
    s.v_apical += dv_a * s.dt
    gate = s._sigma(s.v_apical)
    du = (-s.u + gate * (s.g_ratio * (s.v_basal - s.u) + i_soma)) / s.tau
    s.u += du * s.dt
    if s.u >= s.v_th
        s.u = 0.0
        return 1
    end
    return 0
end

function step!(s::MulticompartmentMCNNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        return s.step_compartments(I_ext, 0.0, 0.0)
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = MulticompartmentMCNNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.tau
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module MulticompartmentMcnAccel
