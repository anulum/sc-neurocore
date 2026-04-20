# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for plif

module PlifAccel

export step!, simulate, ParametricLIFNeuronState

mutable struct ParametricLIFNeuronState
    v::Float64
    a::Float64
    threshold::Float64
    dt::Float64
end

function ParametricLIFNeuronState()
    ParametricLIFNeuronState(0.0, 0.0, 1.0, 1.0)
end

function alpha(s::ParametricLIFNeuronState)
    return 1.0 / (1.0 + exp(-s.a))
end

function step!(s::ParametricLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        spike = (s.v >= s.threshold) ? 1 : 0
        s.v = s.alpha * s.v * (1 - spike) + I_ext
        return (s.v >= s.threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ParametricLIFNeuronState()
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

end # module PlifAccel
