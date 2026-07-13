# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Fourcaud-Trocmé ExpIF recurrence

module ExpifAccel

export ExpIFNeuronState, reset!, simulate, simulate_trace, step!

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
        state.v_rest < state.v_threshold && state.v_reset < state.v_threshold
end

function _rhs(state::ExpIFNeuronState, v::Float64, current::Float64)
    bounded_v = min(v, state.v_threshold)
    exp_term = state.delta_t * exp((bounded_v - state.v_rh) / state.delta_t)
    rhs = (-(bounded_v - state.v_rest) + exp_term + current) / state.tau
    isfinite(rhs) || throw(DomainError(rhs, "ExpIF RK4 derivative must remain finite"))
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
    k2 = _rhs(state, state.v + 0.5 * dt * k1, current)
    k3 = _rhs(state, state.v + 0.5 * dt * k2, current)
    k4 = _rhs(state, state.v + dt * k3, current)
    next_v = state.v + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    isfinite(next_v) || throw(DomainError(next_v, "ExpIF RK4 update must remain finite"))

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

function simulate_trace(
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
    )
    _valid(state, dt) || throw(DomainError(v, "ExpIF state parameters are invalid"))
    isfinite(current) || throw(DomainError(current, "ExpIF input current must be finite"))
    trace = Vector{Float64}(undef, n_steps)
    spikes = 0
    for index in eachindex(trace)
        spikes += step!(state, current)
        trace[index] = state.v
    end
    (trace = trace, spikes = spikes, vf = state.v, rf = state.refractory_remaining)
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
        n_steps,
        current,
    )
    result.trace, result.spikes
end

end
