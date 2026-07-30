# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia source MAT* kernel

"""Source-faithful non-resetting MAT* adaptive-threshold implementation."""
module MatAccel

export MATNeuronState, reset!, simulate, step!, threshold, valid_state

const V_MIN = -200.0
const V_MAX = 200.0
const THETA_MAX = 1.0e9

"""
Complete MAT* state and configuration.

Voltage is relative to rest and never resets. Units are millivolts,
milliseconds, nanoamps, and megaohms. Defaults select the regular-spiking
example of Kobayashi, Tsubo, and Shinomoto (2009).
"""
mutable struct MATNeuronState
    v::Float64
    theta1::Float64
    theta2::Float64
    refractory_remaining::Float64
    omega::Float64
    tau_m::Float64
    tau_1::Float64
    tau_2::Float64
    alpha_1::Float64
    alpha_2::Float64
    resistance::Float64
    refractory_period::Float64
    dt::Float64
end

"""Construct the paper's regular-spiking example profile."""
function MATNeuronState()
    MATNeuronState(0.0, 0.0, 0.0, 0.0, 19.0, 5.0, 10.0, 200.0, 37.0, 2.0, 50.0, 2.0, 0.001)
end

"""Return the instantaneous adaptive threshold in millivolts."""
threshold(state::MATNeuronState)::Float64 = state.omega + state.theta1 + state.theta2

"""Return whether the complete state and configuration are valid."""
function valid_state(state::MATNeuronState)::Bool
    values = (
        state.v,
        state.theta1,
        state.theta2,
        state.refractory_remaining,
        state.omega,
        state.tau_m,
        state.tau_1,
        state.tau_2,
        state.alpha_1,
        state.alpha_2,
        state.resistance,
        state.refractory_period,
        state.dt,
    )
    return all(isfinite, values) &&
           V_MIN <= state.v <= V_MAX &&
           -THETA_MAX <= state.omega <= THETA_MAX &&
           0.0 <= state.theta1 <= THETA_MAX &&
           0.0 <= state.theta2 <= THETA_MAX &&
           0.0 <= state.alpha_1 <= THETA_MAX &&
           0.0 <= state.alpha_2 <= THETA_MAX &&
           state.tau_m > 0.0 &&
           state.tau_1 > 0.0 &&
           state.tau_2 > 0.0 &&
           state.resistance > 0.0 &&
           state.refractory_period >= 0.0 &&
           state.dt > 0.0 &&
           0.0 <= state.refractory_remaining <= state.refractory_period
end

"""
Advance one atomic source MAT* step and return `1` on an event.

Voltage uses forward Euler and exact threshold-history decay. Voltage is never
reset. Invalid input/state returns `-1` without mutation.
"""
function step!(state::MATNeuronState, current::Float64=0.0)::Int
    if !isfinite(current) || !valid_state(state)
        return -1
    end
    v = state.v + state.dt * (-state.v + state.resistance * current) / state.tau_m
    theta1 = state.theta1 * exp(-state.dt / state.tau_1)
    theta2 = state.theta2 * exp(-state.dt / state.tau_2)
    refractory = max(0.0, state.refractory_remaining - state.dt)
    if !(all(isfinite, (v, theta1, theta2, refractory)) &&
         V_MIN <= v <= V_MAX &&
         0.0 <= theta1 <= THETA_MAX &&
         0.0 <= theta2 <= THETA_MAX)
        return -1
    end
    spike = refractory == 0.0 && v >= state.omega + theta1 + theta2
    if spike
        theta1 += state.alpha_1
        theta2 += state.alpha_2
        refractory = state.refractory_period
        if theta1 > THETA_MAX || theta2 > THETA_MAX
            return -1
        end
    end
    state.v = v
    state.theta1 = theta1
    state.theta2 = theta2
    state.refractory_remaining = refractory
    return Int(spike)
end

"""Reset dynamic state while preserving the configured profile."""
function reset!(state::MATNeuronState)::Nothing
    state.v = 0.0
    state.theta1 = 0.0
    state.theta2 = 0.0
    state.refractory_remaining = 0.0
    return nothing
end

"""Run a complete current trace and return all state traces and event outputs."""
function simulate(currents::AbstractVector{<:Real}; state::MATNeuronState=MATNeuronState())
    steps = length(currents)
    voltages = Vector{Float64}(undef, steps)
    theta1 = Vector{Float64}(undef, steps)
    theta2 = Vector{Float64}(undef, steps)
    refractory = Vector{Float64}(undef, steps)
    events = Vector{Int}(undef, steps)
    for index in eachindex(currents)
        event = step!(state, Float64(currents[index]))
        if event < 0
            throw(ArgumentError("invalid MAT step at index $index"))
        end
        voltages[index] = state.v
        theta1[index] = state.theta1
        theta2[index] = state.theta2
        refractory[index] = state.refractory_remaining
        events[index] = event
    end
    return (; voltages, theta1, theta2, refractory, events, state)
end

"""Run `n_steps` under constant current; retained for service compatibility."""
function simulate(n_steps::Int=1000; I_ext::Float64=0.5, dt::Float64=0.001)
    state = MATNeuronState()
    state.dt = dt
    result = simulate(fill(I_ext, n_steps); state=state)
    return result.voltages, sum(result.events)
end

end # module MatAccel
