# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for iqif

module IqifAccel

export step!, simulate, IntegerQIFNeuronState

mutable struct IntegerQIFNeuronState
    v::Float64
    k::Float64
    v_threshold::Float64
    v_reset::Float64
    v_min::Float64
end

function IntegerQIFNeuronState()
    IntegerQIFNeuronState(0.0, 6.0, 1024.0, -1024.0, -2048.0)
end

function step!(s::IntegerQIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.v = max(s.v_min, s.v + (s.v * s.v >> s.k) + I_ext)
        if s.v >= s.v_threshold
            s.v = s.v_reset
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = IntegerQIFNeuronState()
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

end # module IqifAccel
