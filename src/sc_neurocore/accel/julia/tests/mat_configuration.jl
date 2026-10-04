# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source MAT reset, admission and caller-state contracts

using Test
include(joinpath(@__DIR__, "..", "neurons", "mat.jl"))
using .MatAccel

state_bits(state) = Tuple(reinterpret(UInt64, getproperty(state, field)) for field in fieldnames(MATNeuronState))

@testset "Source MAT complete state contracts" begin
    @testset "Empty-vector complete admission" begin
        for field in fieldnames(MATNeuronState), bad in (NaN, Inf, -Inf)
            state = MATNeuronState()
            setproperty!(state, field, bad)
            before = state_bits(state)
            @test_throws ArgumentError simulate(Float64[]; state=state)
            @test step!(state, 0.7) == -1
            @test state_bits(state) == before
        end
    end

    @testset "Atomic zero-rest candidate and recovery" begin
        invalid = [
            (:omega, -1.0e9 - 1.0), (:omega, 1.0e9 + 1.0),
            (:alpha_1, -1.0), (:alpha_1, 1.0e9 + 1.0),
            (:alpha_2, -1.0), (:alpha_2, 1.0e9 + 1.0),
            (:tau_m, 0.0), (:tau_m, -1.0), (:tau_1, 0.0), (:tau_1, -1.0),
            (:tau_2, 0.0), (:tau_2, -1.0), (:resistance, 0.0),
            (:resistance, -1.0), (:refractory_period, -1.0),
            (:dt, 0.0), (:dt, -1.0),
        ]
        for field in fieldnames(MATNeuronState)[5:end], bad in (NaN, Inf, -Inf)
            push!(invalid, (field, bad))
        end
        for (field, bad) in invalid
            state = MATNeuronState()
            state.v = 20.0; state.theta1 = 2.0; state.theta2 = 3.0
            state.refractory_remaining = 0.5
            setproperty!(state, field, bad)
            before = state_bits(state)
            @test_throws ArgumentError reset!(state)
            @test state_bits(state) == before
            setproperty!(state, field, getproperty(MATNeuronState(), field))
            @test reset!(state) === nothing
            @test (state.v, state.theta1, state.theta2, state.refractory_remaining) == (0.0, 0.0, 0.0, 0.0)
            @test step!(state, 0.0) == 0
        end
        for field in (:v, :theta1, :theta2, :refractory_remaining), bad in (NaN, Inf, -Inf, -1.0e308, 1.0e308)
            state = MATNeuronState()
            state.tau_m = 8.0; state.alpha_1 = 7.0; state.refractory_period = 0.0
            setproperty!(state, field, bad)
            before = state_bits(state)
            @test reset!(state) === nothing
            @test (state.v, state.theta1, state.theta2, state.refractory_remaining) == (0.0, 0.0, 0.0, 0.0)
            @test state_bits(state)[5:end] == before[5:end]
            @test step!(state, 0.0) == 0
        end
    end

    @testset "Complete batch refusal preserves caller state" begin
        for currents in ([0.7, NaN], [0.7, Inf], [0.7, -Inf], [0.7, 1.0e308])
            state = MATNeuronState()
            before = state_bits(state)
            @test_throws ArgumentError simulate(currents; state=state)
            @test state_bits(state) == before
            result = simulate([0.7, 0.5]; state=state)
            reference = simulate([0.7, 0.5])
            @test result.state === state
            @test result.voltages == reference.voltages
            @test result.theta1 == reference.theta1 && result.theta2 == reference.theta2
            @test result.refractory == reference.refractory && result.events == reference.events
        end
    end

    @testset "Constant current admission and configured trajectories" begin
        for current in (NaN, Inf, -Inf), count in (0, 1)
            @test_throws ArgumentError simulate(count; I_ext=current)
        end
        for dt in (NaN, Inf, -Inf, 0.0, -1.0), count in (0, 1)
            @test_throws ArgumentError simulate(count; dt=dt)
        end
        @test simulate(0) == (Float64[], 0)
        @test_throws ArgumentError simulate(-1)
        for dt in (0.001, 0.05, 0.5), current in (0.0, 0.5, 0.7)
            trace, events = simulate(256; I_ext=current, dt=dt)
            state = MATNeuronState()
            state.dt = dt
            reference = simulate(fill(current, 256); state=state)
            @test length(trace) == 256
            @test trace == reference.voltages
            @test events == sum(reference.events)
        end
    end
end
