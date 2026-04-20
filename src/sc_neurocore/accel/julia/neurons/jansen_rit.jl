# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for jansen_rit

module JansenRitAccel

export step!, simulate, JansenRitUnitState

mutable struct JansenRitUnitState
    y0::Float64
    y3::Float64
    y1::Float64
    y4::Float64
    y2::Float64
    y5::Float64
    a_exc::Float64
    b_exc::Float64
    a_rate::Float64
    b_rate::Float64
    c::Float64
    e0::Float64
    v0::Float64
    r::Float64
    dt::Float64
end

function JansenRitUnitState()
    JansenRitUnitState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.25, 22.0, 100.0, 50.0, 135.0, 2.5, 6.0, 0.56, 0.001)
end

function _sigmoid(s::JansenRitUnitState, x)
    return 2.0 * s.e0 / (1.0 + exp(s.r * (s.v0 - x)))
end

function step!(s::JansenRitUnitState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s1 = s._sigmoid(s.y1 - s.y2)
        s0 = s._sigmoid(s.c * 0.8 * s.y0)
        s2 = s._sigmoid(s.c * 0.25 * s.y0)
        dy0 = s.y3
        dy3 = s.a_exc * s.a_rate * s1 - 2.0 * s.a_rate * s.y3 - s.a_rate ^ 2 * s.y0
        dy1 = s.y4
        dy4 = s.a_exc * s.a_rate * (p_ext + s.c * 0.8 * s0) - 2.0 * s.a_rate * s.y4 - s.a_rate ^ 2 * s.y1
        dy2 = s.y5
        dy5 = s.b_exc * s.b_rate * s.c * 0.25 * s2 - 2.0 * s.b_rate * s.y5 - s.b_rate ^ 2 * s.y2
        s.y0 += dy0 * s.dt
        s.y3 += dy3 * s.dt
        s.y1 += dy1 * s.dt
        s.y4 += dy4 * s.dt
        s.y2 += dy2 * s.dt
        s.y5 += dy5 * s.dt
        return s.y1 - s.y2
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = JansenRitUnitState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.y0
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module JansenRitAccel
