# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia AdEx parity and fail-closed tests

using Test

include(joinpath(@__DIR__, "neurons", "adex.jl"))
using .AdexAccel

@testset "AdEx Julia parity" begin
    for (current, expected) in ((0.0, 0), (200.0, 4), (500.0, 12))
        _, spikes = simulate(1000; I_ext=current, dt=0.1)
        @test spikes == expected
    end

    state = AdExNeuronState()
    state.v = -60.0
    state.w = 3.0
    current = 250.0
    arg = clamp((state.v - state.v_rh) / state.delta_t, -20.0, 20.0)
    exp_term = state.delta_t * exp(arg)
    expected_v = state.v + (
        (-(state.v - state.v_rest) + exp_term) / state.tau +
        (-state.w + current) / state.c_m
    ) * state.dt
    expected_w = state.w + (
        state.a * (state.v - state.v_rest) - state.w
    ) / state.tau_w * state.dt
    @test step!(state, current) == 0
    @test state.v == expected_v
    @test state.w == expected_w
end

@testset "AdEx Julia fail-closed" begin
    state = AdExNeuronState()
    before = (state.v, state.w, state.dt)
    @test_throws DomainError step!(state, Inf; dt=0.2)
    @test (state.v, state.w, state.dt) == before

    state.dt = 1.0e308
    before = (state.v, state.w, state.dt)
    @test_throws DomainError step!(state, 1.0e308)
    @test (state.v, state.w, state.dt) == before

    state = AdExNeuronState()
    state.v_rest = -63.0
    state.dt = 0.2
    state.a = 0.75
    state.v = -51.0
    state.w = 9.0
    reset!(state)
    @test (state.v, state.w) == (-63.0, 0.0)
    @test (state.dt, state.a) == (0.2, 0.75)
end

@testset "AdEx Julia complete simulation surface" begin
    result = simulate_complete(
        -60.0,
        3.0,
        -65.0,
        -68.0,
        -50.0,
        -55.0,
        2.0,
        20.0,
        100.0,
        0.5,
        7.0,
        200.0,
        0.2,
        200,
        500.0,
    )
    @test length(result.v_trace) == 200
    @test length(result.w_trace) == 200
    @test length(result.events) == 200
    @test sum(result.events) == result.spikes
    @test result.spikes > 0
    @test isfinite(result.vf)
    @test isfinite(result.wf)
    @test (result.vf, result.wf) == (result.v_trace[end], result.w_trace[end])

    empty = simulate_complete(
        -60.0,
        3.0,
        -65.0,
        -68.0,
        -50.0,
        -55.0,
        2.0,
        20.0,
        100.0,
        0.5,
        7.0,
        200.0,
        0.2,
        0,
        500.0,
    )
    @test isempty(empty.v_trace)
    @test isempty(empty.w_trace)
    @test isempty(empty.events)
    @test (empty.spikes, empty.vf, empty.wf) == (0, -60.0, 3.0)
    @test_throws DomainError simulate_complete(
        -65.0,
        0.0,
        -65.0,
        -68.0,
        -50.0,
        -55.0,
        2.0,
        20.0,
        100.0,
        0.5,
        7.0,
        200.0,
        0.1,
        -1,
        0.0,
    )
end
