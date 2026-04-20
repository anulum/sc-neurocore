# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for ilif

module IlifAccel

export step!, simulate, InhibitoryLIFNeuronState

mutable struct InhibitoryLIFNeuronState
    v::Float64
    inh_trace::Float64
    tau_m::Float64
    tau_inh::Float64
    v_threshold::Float64
    v_reset::Float64
    inh_strength::Float64
    dt::Float64
    alpha_m::Float64
    alpha_inh::Float64
end

function InhibitoryLIFNeuronState()
    InhibitoryLIFNeuronState(0.0, 0.0, 10.0, 5.0, 1.0, 0.0, 0.5, 1.0, 0.0, 0.0)
end

function step!(s::InhibitoryLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.inh_trace *= s.alpha_inh
        s.v = s.alpha_m * s.v + I_ext - s.inh_strength * s.inh_trace
        if s.v >= s.v_threshold
            s.v = s.v_reset
            s.inh_trace += 1.0
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = InhibitoryLIFNeuronState()
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

end # module IlifAccel
