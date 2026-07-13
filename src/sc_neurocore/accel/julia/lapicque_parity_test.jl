# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Lapicque native parity tests

using Test

include(joinpath(@__DIR__, "neurons", "lapicque.jl"))
using .LapicqueAccel

@testset "Lapicque source defaults and exact flow" begin
    state = LapicqueNeuronState()
    @test state.v == 0.0
    @test state.v_threshold == 1.0
    @test state.tau == 20.0
    @test state.resistance == 1.0
    @test state.dt == 1.0

    state.v = 0.25
    state.dt = 5.0
    current = 0.5
    v0 = state.v
    v_inf = state.v_rest + state.resistance * current
    expected = v_inf + (v0 - v_inf) * exp(-state.dt / state.tau)
    euler = v0 + (-(v0 - state.v_rest) + state.resistance * current) / state.tau * state.dt
    @test step!(state, current) == 0
    @test state.v ≈ expected atol = 1.0e-15
    @test abs(state.v - euler) > 1.0e-4
end

@testset "Lapicque event goldens and full contract" begin
    for (current, expected_spikes) in ((0.0, 0), (0.5, 0), (2.0, 71), (5.0, 200), (20.0, 500))
        trace, spikes = simulate(1000; I_ext=current)
        @test length(trace) == 1000
        @test spikes == expected_spikes
        @test all(isfinite, trace)
    end

    result = simulate_trace(0.25, -0.1, -0.2, 1.3, 7.5, 1.7, 0.37, 300, 2.2)
    @test result.spikes == 27
    @test result.vf ≈ 0.7838562764025099 atol = 1.0e-15
    @test result.trace[end] == result.vf

    empty = simulate_trace(0.25, -0.1, -0.2, 1.3, 7.5, 1.7, 0.37, 0, 2.2)
    @test isempty(empty.trace)
    @test empty.spikes == 0
    @test empty.vf == 0.25
end

@testset "Lapicque rejection and reset boundaries" begin
    invalid_input = LapicqueNeuronState()
    before = invalid_input.v
    @test_throws DomainError step!(invalid_input, Inf)
    @test invalid_input.v == before

    invalid_state = LapicqueNeuronState()
    invalid_state.tau = 0.0
    @test_throws DomainError step!(invalid_state, 1.0)
    @test invalid_state.v == 0.0

    invalid_update = LapicqueNeuronState()
    invalid_update.v_threshold = 1.0e308
    invalid_update.resistance = 1.0e308
    @test_throws DomainError step!(invalid_update, 1.0e308)
    @test invalid_update.v == 0.0

    explicit_dt = LapicqueNeuronState()
    @test step!(explicit_dt, 0.5; dt=0.25) == 0
    @test explicit_dt.dt == 1.0
    @test_throws DomainError step!(explicit_dt, 0.5; dt=0.0)
    @test explicit_dt.dt == 1.0

    reset_state = LapicqueNeuronState(0.5, -0.25, -0.5, 2.0, 10.0, 2.0, 0.25)
    reset!(reset_state)
    @test reset_state.v == -0.25
    @test reset_state.v_reset == -0.5
    @test reset_state.v_threshold == 2.0
    @test reset_state.tau == 10.0
    @test reset_state.resistance == 2.0
    @test reset_state.dt == 0.25

    @test_throws ArgumentError simulate_trace(0.0, 0.0, 0.0, 1.0, 20.0, 1.0, 1.0, -1, 0.0)
    @test_throws DomainError simulate_trace(0.0, 0.0, 0.0, 1.0, 20.0, 1.0, 1.0, 0, Inf)
    @test_throws DomainError simulate_trace(1.0, 0.0, 0.0, 1.0, 20.0, 1.0, 1.0, 1, 0.0)
end
