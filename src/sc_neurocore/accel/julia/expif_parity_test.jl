# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia ExpIF native parity tests

using Test

include(joinpath(@__DIR__, "neurons", "expif.jl"))
using .ExpifAccel

@testset "ExpIF source defaults" begin
    state = ExpIFNeuronState()
    @test state.v == -65.0
    @test state.v_threshold == 30.0
    @test state.v_rh == -59.9
    @test state.delta_t == 3.48
    @test state.tau == 10.0
    @test state.dt == 0.02
end

@testset "ExpIF RK4 and event goldens" begin
    state = ExpIFNeuronState()
    state.v = -62.0
    state.dt = 0.05
    current = 5.0
    bounded(v) = min(v, state.v_threshold)
    rhs(v) = (
        -(bounded(v) - state.v_rest) +
        state.delta_t * exp((bounded(v) - state.v_rh) / state.delta_t) + current
    ) / state.tau
    k1 = rhs(state.v)
    k2 = rhs(state.v + 0.5 * state.dt * k1)
    k3 = rhs(state.v + 0.5 * state.dt * k2)
    k4 = rhs(state.v + state.dt * k3)
    expected = state.v + (state.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    @test step!(state, current) == 0
    @test state.v ≈ expected atol = 1.0e-12

    for (drive, expected_spikes) in ((0.0, 0), (5.0, 0), (20.0, 2))
        trace, spikes = simulate(1000, drive)
        @test length(trace) == 1000
        @test spikes == expected_spikes
        @test all(isfinite, trace)
    end
end

@testset "ExpIF refractory and fail-closed boundaries" begin
    state = ExpIFNeuronState()
    state.refractory_period = 1.7
    while step!(state, 50.0) == 0 end
    @test state.v == state.v_reset
    @test state.refractory_remaining == 1.7
    for _ in 1:10
        @test step!(state, 50.0) == 0
        @test state.v == state.v_reset
    end
    @test state.refractory_remaining ≈ 1.5 atol = 1.0e-12
    reset!(state)
    @test state.v == state.v_rest
    @test state.refractory_remaining == 0.0

    invalid_input = ExpIFNeuronState()
    before = invalid_input.v
    @test_throws DomainError step!(invalid_input, Inf)
    @test invalid_input.v == before

    invalid_state = ExpIFNeuronState()
    invalid_state.refractory_remaining = 1.0
    @test_throws DomainError step!(invalid_state, 0.0)
    @test invalid_state.refractory_remaining == 1.0

    invalid_update = ExpIFNeuronState()
    invalid_update.dt = 1.0e308
    before_update = invalid_update.v
    @test_throws DomainError step!(invalid_update, 1.0e308)
    @test invalid_update.v == before_update

    @test_throws ArgumentError simulate_trace(
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
        -1,
        0.0,
    )
end
