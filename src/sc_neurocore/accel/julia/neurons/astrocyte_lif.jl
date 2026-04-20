# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for astrocyte_lif

module AstrocyteLifAccel

export step!, simulate, AstrocyteLIFNeuronState

mutable struct AstrocyteLIFNeuronState
    tau_m::Float64
    tau_ca::Float64
    e_l::Float64
    theta::Float64
    v_reset::Float64
    ca_delta::Float64
    ca_thresh::Float64
    g_glio::Float64
    dt::Float64
    v::Float64
    ca::Float64
end

function AstrocyteLIFNeuronState()
    AstrocyteLIFNeuronState(20.0, 500.0, -65.0, -50.0, -65.0, 0.1, 0.5, 2.0, 0.1, -65.0, 0.0)
end

function step_with_pre(s::AstrocyteLIFNeuronState, i_ext, pre_spike)
    dca = -s.ca / s.tau_ca
    if pre_spike
        dca += s.ca_delta / s.dt
    end
    s.ca += dca * s.dt
    s.ca = max(s.ca, 0.0)
    i_glio = (s.ca > s.ca_thresh) ? s.g_glio : 0.0
    dv = (-(s.v - s.e_l) + i_ext + i_glio) / s.tau_m
    s.v += dv * s.dt
    if s.v >= s.theta
        s.v = s.v_reset
        return 1
    end
    return 0
end

function step!(s::AstrocyteLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        return s.step_with_pre(I_ext, pre_spike=false)
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = AstrocyteLIFNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.tau_m
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module AstrocyteLifAccel
