# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia SC resetting-MAT kernel

"""Historical SC candidate-first RK4 adaptive-threshold recurrence."""
module SCResettingMatAccel

export SCResettingMATNeuronState, reset!, simulate, step!, valid_state

const V_MIN = -200.0
const V_MAX = 100.0
const THETA_MAX = 1.0e9

"""Complete state and configuration for the SC resetting-MAT model."""
mutable struct SCResettingMATNeuronState
    v::Float64
    theta1::Float64
    theta2::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold_base::Float64
    tau_m::Float64
    tau_1::Float64
    tau_2::Float64
    h1::Float64
    h2::Float64
    resistance::Float64
    dt::Float64
end

"""Construct the preserved SC defaults."""
function SCResettingMATNeuronState()
    SCResettingMATNeuronState(-70.0, 0.0, 0.0, -70.0, -70.0, -50.0, 10.0, 10.0, 200.0, 5.0, 3.0, 1.0, 1.0)
end

"""Return whether the complete state and configuration are valid."""
function valid_state(state::SCResettingMATNeuronState)::Bool
    values = (
        state.v,
        state.theta1,
        state.theta2,
        state.v_rest,
        state.v_reset,
        state.v_threshold_base,
        state.tau_m,
        state.tau_1,
        state.tau_2,
        state.h1,
        state.h2,
        state.resistance,
        state.dt,
    )
    return all(isfinite, values) &&
           V_MIN <= state.v <= V_MAX &&
           V_MIN <= state.v_reset <= V_MAX &&
           0.0 <= state.theta1 <= THETA_MAX &&
           0.0 <= state.theta2 <= THETA_MAX &&
           0.0 <= state.h1 <= THETA_MAX &&
           0.0 <= state.h2 <= THETA_MAX &&
           state.tau_m > 0.0 && state.tau_1 > 0.0 && state.tau_2 > 0.0 &&
           state.resistance > 0.0 && state.dt > 0.0
end

function derivatives(state::SCResettingMATNeuronState, v::Float64, theta1::Float64, theta2::Float64, current::Float64)
    return (
        (-(v - state.v_rest) + state.resistance * current) / state.tau_m,
        -theta1 / state.tau_1,
        -theta2 / state.tau_2,
    )
end

function candidate(state::SCResettingMATNeuronState, current::Float64)
    k1 = derivatives(state, state.v, state.theta1, state.theta2, current)
    k2 = derivatives(state, state.v + 0.5 * state.dt * k1[1], state.theta1 + 0.5 * state.dt * k1[2], state.theta2 + 0.5 * state.dt * k1[3], current)
    k3 = derivatives(state, state.v + 0.5 * state.dt * k2[1], state.theta1 + 0.5 * state.dt * k2[2], state.theta2 + 0.5 * state.dt * k2[3], current)
    k4 = derivatives(state, state.v + state.dt * k3[1], state.theta1 + state.dt * k3[2], state.theta2 + state.dt * k3[3], current)
    scale = state.dt / 6.0
    return (
        state.v + scale * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        state.theta1 + scale * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
        state.theta2 + scale * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]),
    )
end

"""Advance one atomic candidate-first RK4/reset step."""
function step!(state::SCResettingMATNeuronState, current::Float64=0.0)::Int
    if !isfinite(current) || !valid_state(state)
        return -1
    end
    v, theta1, theta2 = candidate(state, current)
    if !(all(isfinite, (v, theta1, theta2)) &&
         V_MIN <= v <= V_MAX &&
         0.0 <= theta1 <= THETA_MAX &&
         0.0 <= theta2 <= THETA_MAX)
        return -1
    end
    spike = v >= state.v_threshold_base + theta1 + theta2
    if spike
        theta1 += state.h1
        theta2 += state.h2
        if theta1 > THETA_MAX || theta2 > THETA_MAX
            return -1
        end
        v = state.v_reset
    end
    state.v = v
    state.theta1 = theta1
    state.theta2 = theta2
    return Int(spike)
end

"""Reset dynamic state while preserving configuration."""
function reset!(state::SCResettingMATNeuronState)::Nothing
    state.v = state.v_rest
    state.theta1 = 0.0
    state.theta2 = 0.0
    return nothing
end

"""Run a complete current trace and return all state traces and events."""
function simulate(currents::AbstractVector{<:Real}; state::SCResettingMATNeuronState=SCResettingMATNeuronState())
    steps = length(currents)
    voltages = Vector{Float64}(undef, steps)
    theta1 = Vector{Float64}(undef, steps)
    theta2 = Vector{Float64}(undef, steps)
    events = Vector{Int}(undef, steps)
    for index in eachindex(currents)
        event = step!(state, Float64(currents[index]))
        if event < 0
            throw(ArgumentError("invalid SC resetting-MAT step at index $index"))
        end
        voltages[index] = state.v
        theta1[index] = state.theta1
        theta2[index] = state.theta2
        events[index] = event
    end
    return (; voltages, theta1, theta2, events, state)
end

"""Run `n_steps` under constant current; retained for service compatibility."""
function simulate(n_steps::Int=1000; I_ext::Float64=50.0, dt::Float64=1.0)
    state = SCResettingMATNeuronState()
    state.dt = dt
    result = simulate(fill(I_ext, n_steps); state=state)
    return result.voltages, sum(result.events)
end

end # module SCResettingMatAccel
