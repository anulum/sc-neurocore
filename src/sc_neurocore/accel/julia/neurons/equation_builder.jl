# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for equation_builder

module EquationBuilderAccel

export step!, simulate, EquationNeuronState

mutable struct EquationNeuronState
    v::Float64
    dt::Float64
    threshold::Float64
end

function EquationNeuronState()
    EquationNeuronState(-65.0, 0.1, -50.0)
end

function step!(s::EquationNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.v += (-s.v + I_ext) * dt / 20.0
        spike = s.v >= s.threshold
        if spike
            s.v = -65.0
        end
        return spike ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = EquationNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module EquationBuilderAccel
