# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for ibarz_tanaka_map

module IbarzTanakaMapAccel

export step!, simulate, IbarzTanakaMapNeuronState

mutable struct IbarzTanakaMapNeuronState
    x::Float64
    y::Float64
    alpha::Float64
    beta::Float64
    mu::Float64
    sigma::Float64
    x_threshold::Float64
    x_reset::Float64
end

function IbarzTanakaMapNeuronState()
    IbarzTanakaMapNeuronState(-1.0, -2.5, 3.65, 0.25, 0.0005, -1.6, 3.0, -1.0)
end

function _f(s::IbarzTanakaMapNeuronState, x)
    if x <= 0.0
        return s.alpha / (1.0 - x)
    end
    return s.alpha + s.beta * x
end

function step!(s::IbarzTanakaMapNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        x_new = s._f(s.x) + s.y + I_ext
        y_new = s.y - s.mu * (s.x + 1.0) + s.mu * s.sigma
        s.x = x_new
        s.y = y_new
        if s.x >= s.x_threshold
            s.x = s.x_reset
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = IbarzTanakaMapNeuronState()
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

end # module IbarzTanakaMapAccel
