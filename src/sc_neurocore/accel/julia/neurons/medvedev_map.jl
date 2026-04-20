# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for medvedev_map

module MedvedevMapAccel

export step!, simulate, MedvedevMapNeuronState

mutable struct MedvedevMapNeuronState
    x::Float64
    alpha::Float64
    beta::Float64
    x_threshold::Float64
end

function MedvedevMapNeuronState()
    MedvedevMapNeuronState(0.0, 3.5, 0.5, 0.9)
end

function step!(s::MedvedevMapNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        x_prev = s.x
        if s.x < s.beta
            s.x = s.alpha * s.x + I_ext
        else
            s.x = s.alpha * (1.0 - s.x) + I_ext
        end
        s.x = s.x % 1.0
        return (s.x >= s.x_threshold && x_prev < s.x_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = MedvedevMapNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.x
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module MedvedevMapAccel
