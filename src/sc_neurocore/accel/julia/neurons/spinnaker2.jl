# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for spinnaker2

module Spinnaker2Accel

export step!, simulate, SpiNNaker2NeuronState

mutable struct SpiNNaker2NeuronState
    v::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    decay_mult::Float64
    decay_shift::Float64
    refrac_steps::Float64
    _refrac_count::Float64
end

function SpiNNaker2NeuronState()
    SpiNNaker2NeuronState(0.0, 0.0, 0.0, 1024.0, 243.0, 8.0, 2.0, 0.0)
end

function step!(s::SpiNNaker2NeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        if s._refrac_count > 0
            s._refrac_count -= 1
            return 0
        end
        s.v = ((s.v - s.v_rest) * s.decay_mult >> s.decay_shift) + s.v_rest + I_ext
        if s.v >= s.v_threshold
            s.v = s.v_reset
            s._refrac_count = s.refrac_steps
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = SpiNNaker2NeuronState()
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

end # module Spinnaker2Accel
