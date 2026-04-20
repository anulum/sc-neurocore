# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for cochlear_hair_cell

module CochlearHairCellAccel

export step!, simulate, CochlearHairCellState

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

function p_open(s::CochlearHairCellState, displacement)
    return 1.0 / (1.0 + exp(-(displacement - s.x0) / s.delta))
end

function step!(s::CochlearHairCellState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        po = s.p_open(displacement)
        i_met = s.g_max * po * (s.v - s.e_met)
        dv = (-s.g_l * (s.v - s.e_l) - i_met) / s.cap
        s.v += dv * s.dt
        s.glutamate_release = max(s.v + 60.0, 0.0) / 40.0
        return (s.glutamate_release > 0.5) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = CochlearHairCellState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.g_max
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module CochlearHairCellAccel
