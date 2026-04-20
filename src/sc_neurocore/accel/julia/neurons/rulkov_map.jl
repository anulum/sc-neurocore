# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for rulkov_map

module RulkovMapAccel

export step!, simulate, RulkovMapNeuronState

mutable struct RulkovMapNeuronState
    x::Float64
    y::Float64
    alpha::Float64
    sigma::Float64
    mu::Float64
    x_threshold::Float64
end

function RulkovMapNeuronState()
    RulkovMapNeuronState(-1.0, -3.0, 4.0, -1.6, 0.001, 0.0)
end

function step!(s::RulkovMapNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        x_prev = s.x
        if s.x <= 0
            x_new = s.alpha / (1.0 - s.x) + s.y + I_ext
        elseif s.x < s.alpha + s.y + I_ext
            x_new = s.alpha + s.y + I_ext
        else
            x_new = -1.0
        end
        y_new = s.y - s.mu * (s.x + 1.0) + s.mu * s.sigma
        s.x = x_new
        s.y = y_new
        return (s.x >= s.x_threshold && x_prev < s.x_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = RulkovMapNeuronState()
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

end # module RulkovMapAccel
