# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Perfect Integrator native parity tests

using Test

include(joinpath(@__DIR__, "neurons", "perfect_integrator.jl"))
using .PerfectIntegratorAccel

@testset "Perfect Integrator source goldens" begin
    state = PerfectIntegratorNeuronState()
    @test state.v == 0.0
    @test state.c_m == 1.0
    @test state.v_threshold == 1.0
    @test state.v_reset == 0.0
    @test state.dt == 0.1

    for (current, expected_spikes) in ((0.0, 0), (0.333, 32), (0.7, 66), (2.0, 200), (3.0, 250), (5.0, 500), (20.0, 1000))
        trace, spikes = simulate(1000; I_ext=current)
        @test length(trace) == 1000
        @test spikes == expected_spikes
        @test all(isfinite, trace)
    end
end

@testset "Perfect Integrator complete contract" begin
    result = simulate_trace(0.25, 1.7, 1.3, -0.2, 0.37, 300, 2.2)
    @test result.spikes == 75
    @test result.vf == 0.2788235294117647
    @test result.trace[end] == result.vf

    empty = simulate_trace(0.25, 1.7, 1.3, -0.2, 0.37, 0, 2.2)
    @test isempty(empty.trace)
    @test empty.spikes == 0
    @test empty.vf == 0.25
end

@testset "Perfect Integrator rejection and reset boundaries" begin
    invalid_input = PerfectIntegratorNeuronState()
    before = invalid_input.v
    @test_throws DomainError step!(invalid_input, Inf)
    @test invalid_input.v == before

    invalid_state = PerfectIntegratorNeuronState()
    invalid_state.c_m = 0.0
    @test_throws DomainError step!(invalid_state, 1.0)
    @test invalid_state.v == 0.0

    invalid_update = PerfectIntegratorNeuronState()
    invalid_update.v = 0.25
    invalid_update.v_threshold = 1.0e308
    invalid_update.c_m = 1.0e-308
    @test_throws DomainError step!(invalid_update, 1.0e308)
    @test invalid_update.v == 0.25

    explicit_dt = PerfectIntegratorNeuronState()
    @test step!(explicit_dt, 0.5; dt=0.25) == 0
    @test explicit_dt.dt == 0.1
    @test_throws DomainError step!(explicit_dt, 0.5; dt=0.0)
    @test explicit_dt.dt == 0.1

    reset_state = PerfectIntegratorNeuronState(0.5, 2.0, 3.0, -1.0, 0.05)
    reset!(reset_state)
    @test reset_state.v == -1.0
    @test reset_state.c_m == 2.0
    @test reset_state.v_threshold == 3.0
    @test reset_state.dt == 0.05

    @test_throws ArgumentError simulate_trace(0.0, 1.0, 1.0, 0.0, 0.1, -1, 0.0)
    @test_throws DomainError simulate_trace(0.0, 1.0, 1.0, 0.0, 0.1, 0, Inf)
    @test_throws DomainError simulate_trace(1.0, 1.0, 1.0, 0.0, 0.1, 1, 0.0)
end
