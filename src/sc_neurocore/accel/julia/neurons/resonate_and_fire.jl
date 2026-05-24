# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for resonate_and_fire

module ResonateAndFireAccel

export step!, simulate, ResonateAndFireNeuronState, valid, reset!

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

function valid(s::ResonateAndFireNeuronState)::Bool
    return isfinite(s.x) &&
        isfinite(s.y) &&
        isfinite(s.b) &&
        isfinite(s.omega) && s.omega > 0.0 &&
        isfinite(s.threshold) && s.threshold > 0.0 &&
        isfinite(s.dt) && s.dt > 0.0
end

function step!(s::ResonateAndFireNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)::Int
    s.dt = dt
    if !isfinite(I_ext) || !valid(s)
        return 0
    end

    dx = (s.b * s.x - s.omega * s.y + I_ext) * s.dt
    dy = (s.omega * s.x + s.b * s.y) * s.dt
    next_x = s.x + dx
    next_y = s.y + dy
    radius = hypot(next_x, next_y)
    if !isfinite(dx) || !isfinite(dy) || !isfinite(next_x) || !isfinite(next_y) || !isfinite(radius)
        return 0
    end

    s.x = next_x
    s.y = next_y
    if radius >= s.threshold
        s.x = 0.0
        s.y = 0.0
        return 1
    end
    return 0
end

function reset!(s::ResonateAndFireNeuronState)::Nothing
    s.x = 0.0
    s.y = 0.0
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=2.0, dt::Float64=0.05)
    s = ResonateAndFireNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = hypot(s.x, s.y)
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module ResonateAndFireAccel
