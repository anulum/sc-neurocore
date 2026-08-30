# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Fourcaud-Trocmé ExpIF recurrence

module ExpifAccel

export ExpIFNeuronState, reset!, simulate, simulate_complete, simulate_trace, step!

mutable struct ExpIFNeuronState
    v::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    v_rh::Float64
    delta_t::Float64
    tau::Float64
    dt::Float64
    refractory_period::Float64
    refractory_remaining::Float64
    source_profile::Bool
end

ExpIFNeuronState() = ExpIFNeuronState(
    -65.0,
    -65.0,
    -68.0,
    30.0,
    -59.9,
    3.48,
    10.0,
    0.02,
    0.0,
    0.0,
    false,
)

function _valid(state::ExpIFNeuronState, dt::Float64)
    all(
        isfinite,
        (
            state.v,
            state.v_rest,
            state.v_reset,
            state.v_threshold,
            state.v_rh,
            state.delta_t,
            state.tau,
            dt,
            state.refractory_period,
            state.refractory_remaining,
        ),
    ) && state.delta_t > 0.0 && state.tau > 0.0 && dt > 0.0 &&
        state.refractory_period >= 0.0 && state.refractory_remaining >= 0.0 &&
        state.refractory_remaining <= state.refractory_period &&
        state.v_threshold > state.v_rh && state.v < state.v_threshold &&
        state.v_rest < state.v_threshold && state.v_reset < state.v_threshold &&
        (
            !state.source_profile ||
            (
                state.v_rest == -65.0 &&
                state.v_reset == -68.0 &&
                state.v_threshold == -30.0 &&
                state.v_rh == -59.9 &&
                state.delta_t == 3.48 &&
                state.tau == 10.0 &&
                dt < 0.02 &&
                state.refractory_period == 1.7
            )
        )
end

function _rhs(state::ExpIFNeuronState, v::Float64, current::Float64)
    bounded_v = min(v, state.v_threshold)
    exp_term = state.delta_t * exp((bounded_v - state.v_rh) / state.delta_t)
    rhs = (-(bounded_v - state.v_rest) + exp_term + current) / state.tau
    isfinite(rhs) || throw(DomainError(rhs, "ExpIF derivative must remain finite"))
    rhs
end

function step!(state::ExpIFNeuronState, current::Float64 = 0.0; dt::Float64 = state.dt)
    _valid(state, dt) || throw(DomainError(state.v, "ExpIF state parameters are invalid"))
    isfinite(current) || throw(DomainError(current, "ExpIF input current must be finite"))

    if state.refractory_remaining > 0.0
        next_refractory = max(0.0, state.refractory_remaining - dt)
        state.dt = dt
        state.refractory_remaining = next_refractory
        state.v = state.v_reset
        return 0
    end

    k1 = _rhs(state, state.v, current)
    if state.source_profile
        predictor = state.v + dt * k1
        k2 = _rhs(state, predictor, current)
        next_v = state.v + 0.5 * dt * (k1 + k2)
    else
        k2 = _rhs(state, state.v + 0.5 * dt * k1, current)
        k3 = _rhs(state, state.v + 0.5 * dt * k2, current)
        k4 = _rhs(state, state.v + dt * k3, current)
        next_v = state.v + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    end
    isfinite(next_v) || throw(DomainError(next_v, "ExpIF update must remain finite"))

    state.dt = dt
    if next_v >= state.v_threshold
        state.v = state.v_reset
        state.refractory_remaining = state.refractory_period
        return 1
    end
    state.v = next_v
    0
end

function reset!(state::ExpIFNeuronState)
    state.v = state.v_rest
    state.refractory_remaining = 0.0
    nothing
end

function simulate_complete(
    v::Float64,
    v_rest::Float64,
    v_reset::Float64,
    v_threshold::Float64,
    v_rh::Float64,
    delta_t::Float64,
    tau::Float64,
    dt::Float64,
    refractory_period::Float64,
    refractory_remaining::Float64,
    source_profile::Bool,
    n_steps::Int,
    current::Float64,
)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
    state = ExpIFNeuronState(
        v,
        v_rest,
        v_reset,
        v_threshold,
        v_rh,
        delta_t,
        tau,
        dt,
        refractory_period,
        refractory_remaining,
        source_profile,
    )
    _valid(state, dt) || throw(DomainError(v, "ExpIF state parameters are invalid"))
    isfinite(current) || throw(DomainError(current, "ExpIF input current must be finite"))
    voltage = Vector{Float64}(undef, n_steps)
    refractory = Vector{Float64}(undef, n_steps)
    events = Vector{UInt8}(undef, n_steps)
    spikes = 0
    for index in eachindex(voltage)
        event = step!(state, current)
        spikes += event
        voltage[index] = state.v
        refractory[index] = state.refractory_remaining
        events[index] = UInt8(event)
    end
    (
        voltage = voltage,
        refractory = refractory,
        events = events,
        spikes = spikes,
        vf = state.v,
        rf = state.refractory_remaining,
    )
end

function simulate_trace(args...)
    result = simulate_complete(args...)
    (
        trace = result.voltage,
        spikes = result.spikes,
        vf = result.vf,
        rf = result.rf,
    )
end

function simulate(n_steps::Int, current::Float64 = 0.0)
    state = ExpIFNeuronState()
    result = simulate_trace(
        state.v,
        state.v_rest,
        state.v_reset,
        state.v_threshold,
        state.v_rh,
        state.delta_t,
        state.tau,
        state.dt,
        state.refractory_period,
        state.refractory_remaining,
        state.source_profile,
        n_steps,
        current,
    )
    result.trace, result.spikes
end

end
