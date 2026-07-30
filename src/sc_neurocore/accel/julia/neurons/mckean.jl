# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — source-bound space-clamped McKean system

"""Right-continuous RK4 specialization of the McKean/Tonnelier equations."""
module McKeanAccel

export McKeanNeuronState, simulate, step!, valid

"""Complete state and configuration for the source-bound scalar system."""
mutable struct McKeanNeuronState
    v::Float64
    w::Float64
    a::Float64
    lambda::Float64
    mu::Float64
    b::Float64
    dt::Float64
end

McKeanNeuronState() = McKeanNeuronState(0.0, 0.0, 0.25, 1.0, 1.0, 0.01, 0.1)

"""Return whether state and source constraints lie in the safety envelope."""
function valid(state::McKeanNeuronState)
    values = (state.v, state.w, state.a, state.lambda, state.mu, state.b, state.dt)
    return all(isfinite, values) &&
           abs(state.v) <= 1e6 &&
           abs(state.w) <= 1e6 &&
           state.a > 0 &&
           state.lambda > 0 &&
           state.mu > state.lambda * state.a &&
           state.b > 0 &&
           0 < state.dt <= 1
end

function rhs(state::McKeanNeuronState, v, w, current)
    heaviside = v >= state.a ? 1.0 : 0.0
    return -state.lambda * v + state.mu * heaviside - w + current, state.b * v
end

function candidate(state::McKeanNeuronState, current)
    dt = state.dt
    k1 = rhs(state, state.v, state.w, current)
    k2 = rhs(state, state.v + dt * k1[1] / 2, state.w + dt * k1[2] / 2, current)
    k3 = rhs(state, state.v + dt * k2[1] / 2, state.w + dt * k2[2] / 2, current)
    k4 = rhs(state, state.v + dt * k3[1], state.w + dt * k3[2], current)
    scale = dt / 6
    v = state.v + scale * (k1[1] + 2k2[1] + 2k3[1] + k4[1])
    w = state.w + scale * (k1[2] + 2k2[2] + 2k3[2] + k4[2])
    return v, w
end

"""Advance atomically and return `-1` when the transition is invalid."""
function step!(state::McKeanNeuronState, current::Float64 = 0.0)
    if !valid(state) || !isfinite(current)
        return -1
    end
    previous = state.v
    v, w = candidate(state, current)
    if !(isfinite(v) && isfinite(w) && abs(v) <= 1e6 && abs(w) <= 1e6)
        return -1
    end
    state.v, state.w = v, w
    return previous < state.a <= v ? 1 : 0
end

"""Execute a complete current trace and return states, events, and final state."""
function simulate(
    currents::AbstractVector{<:Real};
    state::McKeanNeuronState = McKeanNeuronState(),
)
    voltages = zeros(length(currents))
    recovery = zeros(length(currents))
    events = zeros(Int, length(currents))
    for index in eachindex(currents)
        events[index] = step!(state, Float64(currents[index]))
        voltages[index] = state.v
        recovery[index] = state.w
    end
    return (; voltages, recovery, events, state)
end

end
