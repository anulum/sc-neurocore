# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for gated_lif

module GatedLifAccel

export step!, simulate, GatedLIFNeuronState

mutable struct GatedLIFNeuronState
    v::Float64
    gate_v::Float64
    gate_i::Float64
    v_threshold::Float64
    dt::Float64
end

function GatedLIFNeuronState()
    GatedLIFNeuronState(0.0, 0.9, 1.0, 1.0, 1.0)
end

function step!(s::GatedLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.v = s.gate_v * s.v + s.gate_i * I_ext
        if s.v >= s.v_threshold
            s.v -= s.v_threshold
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = GatedLIFNeuronState()
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

end # module GatedLifAccel
