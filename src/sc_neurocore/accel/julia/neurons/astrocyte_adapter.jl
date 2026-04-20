# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for astrocyte_adapter

module AstrocyteAdapterAccel

export step!, simulate, AstrocyteNeuronState

mutable struct AstrocyteNeuronState
    ca_threshold::Float64
    dt::Float64
end

function AstrocyteNeuronState()
    AstrocyteNeuronState(0.3, 0.01)
end

function ca(s::AstrocyteNeuronState)
    return s._astro.ca
end

function ip3(s::AstrocyteNeuronState)
    return s._astro.ip3
end

function step!(s::AstrocyteNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        ca = s._astro.step(I_ext)
        s.v = ca
        return (ca > s.ca_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = AstrocyteNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.ca_threshold
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module AstrocyteAdapterAccel
