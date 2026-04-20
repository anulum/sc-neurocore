# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for truenorth

module TruenorthAccel

export step!, simulate, TrueNorthNeuronState

mutable struct TrueNorthNeuronState
    v::Float64
    leak::Float64
    threshold::Float64
    v_reset::Float64
end

function TrueNorthNeuronState()
    TrueNorthNeuronState(0.0, 0.0, 100.0, 0.0)
end

function step!(s::TrueNorthNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.v = s.v + weighted_input - s.leak
        if s.v >= s.threshold
            s.v = s.v_reset
            return 1
        end
        if s.v < -s.threshold
            s.v = s.v_reset
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = TrueNorthNeuronState()
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

end # module TruenorthAccel
