# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for plif

module PlifAccel

export step!, simulate, alpha, valid, reset!, ParametricLIFNeuronState

mutable struct ParametricLIFNeuronState
    v::Float64
    a::Float64
    threshold::Float64
    dt::Float64
end

function ParametricLIFNeuronState()
    ParametricLIFNeuronState(0.0, 0.0, 1.0, 1.0)
end

function alpha(s::ParametricLIFNeuronState)
    if s.a >= 0.0
        z = exp(-s.a)
        return 1.0 / (1.0 + z)
    end
    z = exp(s.a)
    return z / (1.0 + z)
end

function step!(s::ParametricLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !valid(s) || !isfinite(I_ext)
        return 0
    end
    spike = (s.v >= s.threshold) ? 1.0 : 0.0
    next_v = alpha(s) * s.v * (1.0 - spike) + I_ext
    if !isfinite(next_v)
        return 0
    end
    s.v = next_v
    return (next_v >= s.threshold) ? 1 : 0
end

function valid(s::ParametricLIFNeuronState)
    return isfinite(s.v) &&
        isfinite(s.a) &&
        isfinite(s.threshold) &&
        s.threshold > 0.0 &&
        isfinite(s.dt) &&
        s.dt > 0.0
end

function reset!(s::ParametricLIFNeuronState)
    s.v = 0.0
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ParametricLIFNeuronState()
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

end # module PlifAccel
