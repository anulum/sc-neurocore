# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained unit-capacitance respiratory recurrence

module SCUnitCapacitanceRespiratoryAccel

include("butera_respiratory.jl")
using .ButeraRespiratoryAccel

export SCUnitCapacitanceRespiratoryNeuronState, step!, simulate

"""Construct the count-neutral historical SC respiratory profile."""
function SCUnitCapacitanceRespiratoryNeuronState()
    state = ButeraRespiratoryAccel.ButeraRespiratoryNeuronState()
    state.capacitance = 1.0
    state.e_syn = -10.0
    return state
end

"""Advance the retained SC recurrence and return its observational event."""
function step!(state::ButeraRespiratoryAccel.ButeraRespiratoryNeuronState, current::Float64=0.0)
    return ButeraRespiratoryAccel.step!(state, current)
end

"""Run the retained SC profile for a fixed current."""
function simulate(n_steps::Int=1000; current::Float64=20.0)
    state = SCUnitCapacitanceRespiratoryNeuronState()
    events = 0
    for _ in 1:n_steps
        events += step!(state, current)
    end
    return state, events
end

end # module SCUnitCapacitanceRespiratoryAccel
