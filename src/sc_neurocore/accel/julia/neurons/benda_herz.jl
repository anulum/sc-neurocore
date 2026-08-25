# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Benda-Herz adaptation runtime

module BendaHerzAccel

export BendaHerzNeuronState, step!, reset!, valid, simulate

"""State and parameters for Benda-Herz equations (8) and (45)."""
mutable struct BendaHerzNeuronState
    a::Float64
    phase::Float64
    onset_gain::Float64
    rheobase::Float64
    adaptation_slope::Float64
    tau_a::Float64
    dt::Float64
end

BendaHerzNeuronState() = BendaHerzNeuronState(0.0, 0.0, 60.0, 0.0, 0.1, 100.0, 0.1)

function valid(s::BendaHerzNeuronState)::Bool
    isfinite(s.a) && s.a >= 0.0 && isfinite(s.phase) && 0.0 <= s.phase < 1.0 &&
        isfinite(s.onset_gain) && s.onset_gain > 0.0 && isfinite(s.rheobase) &&
        isfinite(s.adaptation_slope) && s.adaptation_slope >= 0.0 &&
        isfinite(s.tau_a) && s.tau_a > 0.0 && isfinite(s.dt) && s.dt > 0.0
end

function rhs(s::BendaHerzNeuronState, a::Float64, current::Float64)
    rate = s.onset_gain * sqrt(max(current - a - s.rheobase, 0.0))
    ((s.adaptation_slope * rate - a) / s.tau_a, rate / 1000.0)
end

"""Advance one candidate-first RK4 sample and return the phase event."""
function step!(s::BendaHerzNeuronState, current::Float64=0.0)::Int
    valid(s) && isfinite(current) || return -1
    k1a, k1p = rhs(s, s.a, current)
    k2a, k2p = rhs(s, s.a + 0.5*s.dt*k1a, current)
    k3a, k3p = rhs(s, s.a + 0.5*s.dt*k2a, current)
    k4a, k4p = rhs(s, s.a + s.dt*k3a, current)
    scale = s.dt / 6.0
    next_a = s.a + scale*(k1a + 2k2a + 2k3a + k4a)
    next_phase = s.phase + scale*(k1p + 2k2p + 2k3p + k4p)
    isfinite(next_a) && next_a >= 0.0 && isfinite(next_phase) && 0.0 <= next_phase < 2.0 || return -1
    s.a = next_a
    if next_phase >= 1.0
        s.phase = 0.0
        return 1
    end
    s.phase = next_phase
    0
end

function reset!(s::BendaHerzNeuronState)::Nothing
    s.a = 0.0
    s.phase = 0.0
    nothing
end

"""Return complete adaptation, phase, and event traces."""
function simulate(currents::AbstractVector{Float64}; state::BendaHerzNeuronState=BendaHerzNeuronState())
    adaptation = Vector{Float64}(undef, length(currents))
    phases = Vector{Float64}(undef, length(currents))
    events = Vector{Int64}(undef, length(currents))
    for index in eachindex(currents)
        events[index] = step!(state, currents[index])
        adaptation[index] = state.a
        phases[index] = state.phase
    end
    (; adaptation, phases, events, state)
end

end
