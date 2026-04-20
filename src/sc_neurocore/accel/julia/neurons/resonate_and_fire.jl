# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for resonate_and_fire

module ResonateAndFireAccel

export step!, simulate, ResonateAndFireNeuronState

mutable struct ResonateAndFireNeuronState
    x::Float64
    y::Float64
    b::Float64
    omega::Float64
    threshold::Float64
    dt::Float64
end

function ResonateAndFireNeuronState()
    ResonateAndFireNeuronState(0.0, 0.0, -0.1, 1.0, 1.0, 0.05)
end

function step!(s::ResonateAndFireNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        dx = (s.b * s.x - s.omega * s.y + I_ext) * s.dt
        dy = (s.omega * s.x + s.b * s.y) * s.dt
        s.x += dx
        s.y += dy
        r = sqrt(s.x ^ 2 + s.y ^ 2)
        if r >= s.threshold
            s.x = 0.0
            s.y = 0.0
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ResonateAndFireNeuronState()
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

end # module ResonateAndFireAccel
