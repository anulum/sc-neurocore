# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for ermentrout_kopell_pop

module ErmentroutKopellPopAccel

export step!, simulate, ErmentroutKopellPopulationState

mutable struct ErmentroutKopellPopulationState
    r::Float64
    v::Float64
    tau::Float64
    delta::Float64
    eta_bar::Float64
    j::Float64
    dt::Float64
end

function ErmentroutKopellPopulationState()
    ErmentroutKopellPopulationState(0.1, -2.0, 1.0, 1.0, -5.0, 15.0, 0.01)
end

function step!(s::ErmentroutKopellPopulationState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        dr = (s.delta / (pi * s.tau) + 2.0 * s.r * s.v) / s.tau * s.dt
        dv = (s.v ^ 2 + s.eta_bar + ext_input + s.j * s.tau * s.r - (pi * s.tau * s.r) ^ 2) / s.tau * s.dt
        s.r = max(0.0, s.r + dr)
        s.v += dv
        return s.r
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ErmentroutKopellPopulationState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.r
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module ErmentroutKopellPopAccel
