# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for pernarowski

module PernarowskiAccel

export step!, simulate, PernarowskiNeuronState

mutable struct PernarowskiNeuronState
    v::Float64
    w::Float64
    z::Float64
    alpha::Float64
    beta::Float64
    eps1::Float64
    eps2::Float64
    gamma::Float64
    dt::Float64
    v_threshold::Float64
end

function PernarowskiNeuronState()
    PernarowskiNeuronState(-1.0, 0.0, 0.0, 0.1, 0.5, 0.1, 0.001, 0.5, 0.1, 0.5)
end

function step!(s::PernarowskiNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        f_v = s.v - s.v ^ 3 / 3.0
        dv = (f_v - s.w - s.z + I_ext) * s.dt
        dw = s.eps1 * (s.v - s.gamma * s.w + s.alpha) * s.dt
        dz = s.eps2 * (s.beta * (s.v + 0.7) - s.z) * s.dt
        s.v += dv
        s.w += dw
        s.z += dz
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = PernarowskiNeuronState()
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

end # module PernarowskiAccel
