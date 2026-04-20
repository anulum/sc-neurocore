# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for courage_nekorkin_map

module CourageNekorkinMapAccel

export step!, simulate, CourageNekorkinMapNeuronState

mutable struct CourageNekorkinMapNeuronState
    x::Float64
    y::Float64
    alpha::Float64
    beta::Float64
    j::Float64
    x_threshold::Float64
end

function CourageNekorkinMapNeuronState()
    CourageNekorkinMapNeuronState(0.0, 0.0, 3.0, 0.001, 0.1, 1.0)
end

function _f(s::CourageNekorkinMapNeuronState, x)
    if x < 0
        return s.alpha * x
    end
    return s.alpha * x / (1.0 + s.alpha * x)
end

function step!(s::CourageNekorkinMapNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        x_prev = s.x
        x_new = s._f(s.x) + s.y + I_ext + s.j
        y_new = s.y - s.beta * (s.x + 1.0)
        s.x = max(min(x_new, 1000000.0), -1000000.0)
        s.y = max(min(y_new, 1000000.0), -1000000.0)
        return (s.x >= s.x_threshold && x_prev < s.x_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = CourageNekorkinMapNeuronState()
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

end # module CourageNekorkinMapAccel
