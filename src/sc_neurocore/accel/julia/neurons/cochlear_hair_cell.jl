# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for cochlear_hair_cell

module CochlearHairCellAccel

export step!, simulate, validate_cochlear_hair_cell, CochlearHairCellState

mutable struct CochlearHairCellState
    g_max::Float64
    e_met::Float64
    g_l::Float64
    e_l::Float64
    cap::Float64
    x0::Float64
    delta::Float64
    dt::Float64
    v::Float64
    glutamate_release::Float64
end

function CochlearHairCellState()
    CochlearHairCellState(10.0, 0.0, 1.0, -60.0, 10.0, 0.0, 0.1, 0.01, -60.0, 0.0)
end

_finite(xs...) = all(isfinite, xs)

function validate_cochlear_hair_cell(s::CochlearHairCellState)::Bool
    return _finite(s.g_max, s.e_met, s.g_l, s.e_l, s.cap, s.x0, s.delta, s.dt, s.v, s.glutamate_release) &&
        s.g_max >= 0.0 && s.g_l > 0.0 && s.cap > 0.0 && s.delta > 0.0 && s.dt > 0.0 && s.glutamate_release >= 0.0
end

function p_open(s::CochlearHairCellState, displacement::Float64)
    (!_finite(displacement, s.x0, s.delta) || s.delta <= 0.0) && return nothing
    z = (displacement - s.x0) / s.delta
    if z >= 0.0
        po = 1.0 / (1.0 + exp(-z))
    else
        ez = exp(z)
        po = ez / (1.0 + ez)
    end
    return isfinite(po) ? po : nothing
end

function step!(s::CochlearHairCellState, displacement::Float64=0.0; dt::Union{Nothing,Float64}=nothing)
    if dt !== nothing
        (!isfinite(dt) || dt <= 0.0) && return -1
        s.dt = dt
    end
    (!validate_cochlear_hair_cell(s) || !isfinite(displacement)) && return -1
    po = p_open(s, displacement)
    po === nothing && return -1
    g_met = s.g_max * po
    g_total = s.g_l + g_met
    (!isfinite(g_total) || g_total <= 0.0) && return -1
    v_inf = (s.g_l * s.e_l + g_met * s.e_met) / g_total
    candidate_v = v_inf + (s.v - v_inf) * exp(-(g_total / s.cap) * s.dt)
    candidate_release = max(candidate_v + 60.0, 0.0) / 40.0
    (!isfinite(candidate_v) || !isfinite(candidate_release)) && return -1
    s.v = candidate_v
    s.glutamate_release = candidate_release
    return (s.glutamate_release > 0.5) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = CochlearHairCellState()
    s.dt = dt
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext)
        trace[t] = s.v
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module CochlearHairCellAccel
