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

function _valid(s::JansenRitUnitState)
    return all(isfinite, (
        s.y0, s.y3, s.y1, s.y4, s.y2, s.y5,
        s.a_exc, s.b_exc, s.a_rate, s.b_rate, s.c, s.e0, s.v0, s.r, s.dt,
    )) &&
        s.a_exc > 0.0 &&
        s.b_exc > 0.0 &&
        s.a_rate > 0.0 &&
        s.b_rate > 0.0 &&
        s.c >= 0.0 &&
        s.e0 > 0.0 &&
        s.r > 0.0 &&
        s.dt > 0.0
end

function _sigmoid(s::JansenRitUnitState, x)
    if !isfinite(x)
        return NaN
    end
    exponent = s.r * (s.v0 - x)
    if exponent >= 0.0
        exp_neg = exp(-exponent)
        return 2.0 * s.e0 * exp_neg / (1.0 + exp_neg)
    end
    return 2.0 * s.e0 / (1.0 + exp(exponent))
end

function step!(s::JansenRitUnitState, I_ext::Float64=220.0; dt::Float64=s.dt)
    if !isfinite(I_ext) || !isfinite(dt) || dt <= 0.0 || !_valid(s)
        throw(ArgumentError("Jansen-Rit input, state, and timestep must be finite and physical"))
    end
    s1 = _sigmoid(s, s.y1 - s.y2)
    s0 = _sigmoid(s, s.c * 0.8 * s.y0)
    s2 = _sigmoid(s, s.c * 0.25 * s.y0)
    dy0 = s.y3
    dy3 = s.a_exc * s.a_rate * s1 - 2.0 * s.a_rate * s.y3 - s.a_rate ^ 2 * s.y0
    dy1 = s.y4
    dy4 = s.a_exc * s.a_rate * (I_ext + s.c * 0.8 * s0) - 2.0 * s.a_rate * s.y4 - s.a_rate ^ 2 * s.y1
    dy2 = s.y5
    dy5 = s.b_exc * s.b_rate * s.c * 0.25 * s2 - 2.0 * s.b_rate * s.y5 - s.b_rate ^ 2 * s.y2

    next_y0 = s.y0 + dy0 * dt
    next_y3 = s.y3 + dy3 * dt
    next_y1 = s.y1 + dy1 * dt
    next_y4 = s.y4 + dy4 * dt
    next_y2 = s.y2 + dy2 * dt
    next_y5 = s.y5 + dy5 * dt
    if !all(isfinite, (next_y0, next_y3, next_y1, next_y4, next_y2, next_y5))
        throw(ArgumentError("Jansen-Rit candidate state became non-finite"))
    end
    s.y0 = next_y0
    s.y3 = next_y3
    s.y1 = next_y1
    s.y4 = next_y4
    s.y2 = next_y2
    s.y5 = next_y5
    return s.y1 - s.y2
end

function simulate(n_steps::Int=1000; I_ext::Float64=220.0, dt::Float64=0.001)
    s = JansenRitUnitState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = result
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module JansenRitAccel
