# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for hindmarsh_rose

module HindmarshRoseAccel

export step!, simulate, HindmarshRoseNeuronState

mutable struct HindmarshRoseNeuronState
    x::Float64
    y::Float64
    z::Float64
    b::Float64
    r::Float64
    s::Float64
    x_rest::Float64
    dt::Float64
    x_threshold::Float64
end

function HindmarshRoseNeuronState()
    HindmarshRoseNeuronState(-1.6, -10.0, 2.0, 3.0, 0.001, 4.0, -1.6, 0.1, 1.0)
end

function step!(s::HindmarshRoseNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        x_prev = s.x
        dx = (s.y - s.x ^ 3 + s.b * s.x ^ 2 - s.z + I_ext) * s.dt
        dy = (1.0 - 5.0 * s.x ^ 2 - s.y) * s.dt
        dz = s.r * (s.s * (s.x - s.x_rest) - s.z) * s.dt
        s.x += dx
        s.y += dy
        s.z += dz
        return (s.x >= s.x_threshold && x_prev < s.x_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = HindmarshRoseNeuronState()
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

end # module HindmarshRoseAccel
