# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for cazelles_map

module CazellesMapAccel

export step!, simulate, CazellesMapNeuronState

mutable struct CazellesMapNeuronState
    x::Float64
    y::Float64
    a::Float64
    epsilon::Float64
    sigma::Float64
    x_threshold::Float64
end

function CazellesMapNeuronState()
    CazellesMapNeuronState(0.1, 0.0, 3.8, 0.01, 0.5, 0.9)
end

function step!(s::CazellesMapNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        f = s.a * s.x * (1.0 - s.x)
        x_new = f - s.y + I_ext
        y_new = s.y + s.epsilon * (s.x - s.sigma)
        s.x = clamp(x_new, -2.0, 2.0)
        s.y = y_new
        return (s.x >= s.x_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = CazellesMapNeuronState()
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

end # module CazellesMapAccel
