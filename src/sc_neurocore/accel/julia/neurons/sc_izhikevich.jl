# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for neurons/sc_izhikevich

module ScIzhikevichAccel

using Statistics, LinearAlgebra

mutable struct SCIzhikevichNeuronState
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    dt::Float64
    noise_std::Float64
    seed::Float64
end

function SCIzhikevichNeuronState()
    SCIzhikevichNeuronState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function step(s::SCIzhikevichNeuronState, input_current)
    # Two half-steps for numerical stability on 0.04v² term.
    # Izhikevich (2003) recommends dt ≤ 0.5 ms; we split each dt into two.
    half_dt = s.dt * 0.5
    for _ in 1:2
        dv = (0.04 * s.v^2 + 5 * s.v + 140 - s.u + input_current) * half_dt
        du = (s.a * (s.b * s.v - s.u)) * half_dt
        s.v += dv
        s.u += du
    if s.noise_std > 0.0
        s.v += float(s._rng.normal(0.0, s.noise_std))
    if s.v >= IZH_SPIKE_THRESHOLD
        spike = 1
        s.v = s.c
        s.u += s.d
    else
        spike = 0
    return spike
end

function reset_state(s::SCIzhikevichNeuronState)
    s.v = s.c  # membrane potential
    s.u = s.b * s.v
end

function get_state(s::SCIzhikevichNeuronState)
    return Dict("v": float(s.v), "u": float(s.u))
end

end # module ScIzhikevichAccel
end
end
end
