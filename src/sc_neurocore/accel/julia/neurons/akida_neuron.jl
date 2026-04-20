# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for akida_neuron

module AkidaNeuronAccel

export step!, simulate, AkidaNeuronState

mutable struct AkidaNeuronState
    v::Float64
    threshold::Float64
    modulation::Float64
    _rank::Float64
    _spiked::Float64
    _current_modulation::Float64
end

function AkidaNeuronState()
    AkidaNeuronState(0.0, 100.0, 0.75, 0.0, 0.0, 1.0)
end

function step!(s::AkidaNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        if weight != 0
            scaled = Int(weight * s._current_modulation)
            s.v += scaled
            s._rank += 1
            s._current_modulation *= s.modulation
        end
        if s.v >= s.threshold && (! s._spiked)
            s._spiked = true
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = AkidaNeuronState()
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

end # module AkidaNeuronAccel
