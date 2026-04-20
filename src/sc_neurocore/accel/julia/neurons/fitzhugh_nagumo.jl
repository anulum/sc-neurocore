# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for fitzhugh_nagumo

module FitzhughNagumoAccel

export step!, simulate, FitzHughNagumoNeuronState

mutable struct FitzHughNagumoNeuronState
    v::Float64
    w::Float64
    a::Float64
    b::Float64
    epsilon::Float64
    dt::Float64
    v_threshold::Float64
end

function FitzHughNagumoNeuronState()
    FitzHughNagumoNeuronState(-1.0, -0.5, 0.7, 0.8, 0.08, 0.1, 1.0)
end

function step!(s::FitzHughNagumoNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        dv = (s.v - s.v ^ 3 / 3.0 - s.w + I_ext) * s.dt
        dw = s.epsilon * (s.v + s.a - s.b * s.w) * s.dt
        s.v += dv
        s.w += dw
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = FitzHughNagumoNeuronState()
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

end # module FitzhughNagumoAccel
