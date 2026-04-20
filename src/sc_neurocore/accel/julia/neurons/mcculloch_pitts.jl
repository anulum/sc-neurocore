# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for mcculloch_pitts

module MccullochPittsAccel

export step!, simulate, McCullochPittsNeuronState

mutable struct McCullochPittsNeuronState
    theta::Float64
end

function McCullochPittsNeuronState()
    McCullochPittsNeuronState(1.0)
end

function step!(s::McCullochPittsNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        return (weighted_input >= s.theta) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = McCullochPittsNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.theta
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module MccullochPittsAccel
