# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for astrocyte_lif

module AstrocyteLifAccel

export step!, step_with_pre!, simulate, validate, AstrocyteLIFNeuronState

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

function validate(s::AstrocyteLIFNeuronState)::Bool
    all(x -> isfinite(x) && x > 0.0, (s.tau_m, s.tau_ca, s.dt)) &&
        all(isfinite, (s.e_l, s.theta, s.v_reset, s.v)) &&
        s.theta > s.v_reset &&
        all(x -> isfinite(x) && x >= 0.0, (s.ca_delta, s.ca_thresh, s.g_glio, s.ca))
end

function step_with_pre!(s::AstrocyteLIFNeuronState, i_ext::Float64, pre_spike::Bool)
    if !validate(s) || !isfinite(i_ext)
        return -1
    end
    dca = -s.ca / s.tau_ca
    if pre_spike
        dca += s.ca_delta / s.dt
    end
    ca_next = max(s.ca + dca * s.dt, 0.0)
    if !isfinite(ca_next) || ca_next < 0.0
        return -1
    end
    i_glio = (ca_next > s.ca_thresh) ? s.g_glio : 0.0
    dv = (-(s.v - s.e_l) + i_ext + i_glio) / s.tau_m
    v_next = s.v + dv * s.dt
    if !isfinite(v_next)
        return -1
    end
    s.ca = ca_next
    if v_next >= s.theta
        s.v = s.v_reset
        return 1
    end
    s.v = v_next
    return 0
end

function step!(s::AstrocyteLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    return step_with_pre!(s, I_ext, false)
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = AstrocyteLIFNeuronState()
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

end # module AstrocyteLifAccel
