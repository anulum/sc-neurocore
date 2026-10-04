# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained adaptive LIF Julia configuration contracts

using Test
include(joinpath(@__DIR__, "..", "neurons", "sc_non_resetting_adaptive_lif.jl"))
using .SCNonResettingAdaptiveLifAccel

@testset "SC adaptive LIF complete invalid configuration" begin
    for field in fieldnames(SCNonResettingAdaptiveLIFNeuronState), bad in (NaN, Inf, -Inf)
        state = SCNonResettingAdaptiveLIFNeuronState()
        setproperty!(state, field, bad)
        before = (reinterpret(UInt64, state.v), reinterpret(UInt64, state.theta))
        @test !valid(state)
        @test_throws DomainError step!(state, 20.0)
        @test_throws DomainError simulate(Float64[]; state=state)
        @test (reinterpret(UInt64, state.v), reinterpret(UInt64, state.theta)) == before
    end
    for (field, value) in ((:delta_theta, -1.0), (:r_m, -1.0), (:tau_m, 0.0), (:tau_theta, 0.0), (:dt, 0.0))
        state = SCNonResettingAdaptiveLIFNeuronState()
        setproperty!(state, field, value)
        @test_throws DomainError simulate(Float64[]; state=state)
        @test_throws DomainError step!(state, 20.0)
    end
end

@testset "SC adaptive LIF constant-current admission and full vector parity" begin
    for current in (NaN, Inf, -Inf), n_steps in (0, 1)
        @test_throws DomainError simulate(n_steps; current=current)
    end
    for dt in (NaN, Inf, -Inf, 0.0, -1.0), n_steps in (0, 1)
        @test_throws DomainError simulate(n_steps; dt=dt)
    end
    @test simulate(0) == (Float64[], 0)
    @test_throws ArgumentError simulate(-1)
    for dt in (0.1, 10.0, 40.0, 1000.0, 1e308), current in (0.0, 20.0)
        trace, spikes = simulate(256; current=current, dt=dt)
        state = SCNonResettingAdaptiveLIFNeuronState()
        state.dt = dt
        reference = simulate(fill(current, 256); state=state)
        @test trace == reference.voltages
        @test spikes == sum(reference.events)
    end
end

@testset "SC adaptive LIF atomic reset and recovery" begin
    for field in (:v_rest, :theta_rest, :delta_theta, :tau_m, :tau_theta, :r_m, :dt), bad in (NaN, Inf, -Inf)
        state = SCNonResettingAdaptiveLIFNeuronState()
        state.v = -64.0; state.theta = -49.0
        setproperty!(state, field, bad)
        @test_throws DomainError reset!(state)
        @test (state.v, state.theta) == (-64.0, -49.0)
        setproperty!(state, field, getproperty(SCNonResettingAdaptiveLIFNeuronState(), field))
        state.v = NaN; state.theta = Inf
        @test reset!(state) === nothing
        @test (state.v, state.theta) == (-65.0, -50.0)
    end
end

@testset "SC adaptive LIF finite overflow and accepted large timestep" begin
    state = SCNonResettingAdaptiveLIFNeuronState()
    state.r_m = 1e308
    @test_throws DomainError step!(state, 20.0)
    @test (state.v, state.theta) == (-65.0, -50.0)
    @test step!(state, 0.0) == 0
    state.v = 1.5e308; state.theta = 1e308
    state.theta_rest = 1e308; state.delta_theta = 1e308
    @test_throws DomainError step!(state, 0.0)
    @test (state.v, state.theta) == (1.5e308, 1e308)
    for dt in (0.1, 10.0, 40.0, 1000.0, 1e308)
        state = SCNonResettingAdaptiveLIFNeuronState()
        state.dt = dt
        result = simulate(repeat([20.0, 0.0, 60.0, 10.0], 64); state=state)
        @test length(result.events) == 256
        @test all(isfinite, result.voltages) && all(isfinite, result.theta)
        @test all(x -> x in (0, 1), result.events)
        @test reset!(state) === nothing
        @test (state.v, state.theta) == (-65.0, -50.0)
    end
end
