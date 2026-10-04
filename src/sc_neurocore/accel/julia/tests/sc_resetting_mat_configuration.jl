# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC resetting-MAT configuration and reset contracts

using Test
include(joinpath(@__DIR__, "..", "neurons", "sc_resetting_mat.jl"))
using .SCResettingMatAccel

state_bits(state) = Tuple(reinterpret(UInt64, getproperty(state, field)) for field in fieldnames(SCResettingMATNeuronState))

@testset "SC resetting-MAT configuration contracts" begin
@testset "SC resetting-MAT complete empty-vector admission" begin
    for field in fieldnames(SCResettingMATNeuronState), bad in (NaN, Inf, -Inf)
        state = SCResettingMATNeuronState()
        setproperty!(state, field, bad)
        before = state_bits(state)
        @test_throws ArgumentError simulate(Float64[]; state=state)
        @test step!(state, 50.0) == -1
        @test state_bits(state) == before
    end
end

@testset "SC resetting-MAT atomic resting candidate and recovery" begin
    invalid = [
        (:v_rest, -500.0), (:v_rest, 500.0),
        (:v_reset, -201.0), (:v_reset, 101.0),
        (:tau_m, 0.0), (:tau_m, -1.0), (:tau_1, 0.0), (:tau_1, -1.0),
        (:tau_2, 0.0), (:tau_2, -1.0), (:h1, -1.0), (:h1, 1.0e9 + 1.0),
        (:h2, -1.0), (:h2, 1.0e9 + 1.0), (:resistance, 0.0),
        (:resistance, -1.0), (:dt, 0.0), (:dt, -1.0),
    ]
    for field in fieldnames(SCResettingMATNeuronState)[4:end], bad in (NaN, Inf, -Inf)
        push!(invalid, (field, bad))
    end
    for (field, bad) in invalid
        state = SCResettingMATNeuronState()
        state.v = -65.0; state.theta1 = 2.0; state.theta2 = 3.0
        setproperty!(state, field, bad)
        before = state_bits(state)
        @test_throws ArgumentError reset!(state)
        @test state_bits(state) == before
        setproperty!(state, field, getproperty(SCResettingMATNeuronState(), field))
        @test reset!(state) === nothing
        @test (state.v, state.theta1, state.theta2) == (-70.0, 0.0, 0.0)
        @test step!(state, 0.0) == 0
    end
    for field in (:v, :theta1, :theta2), bad in (NaN, Inf, -Inf, -1.0e308, 1.0e308)
        state = SCResettingMATNeuronState()
        state.v_rest = -65.0; state.tau_m = 12.0; state.h1 = 4.0
        setproperty!(state, field, bad)
        before = state_bits(state)
        @test reset!(state) === nothing
        @test (state.v, state.theta1, state.theta2) == (-65.0, 0.0, 0.0)
        @test state_bits(state)[4:end] == before[4:end]
        @test step!(state, 0.0) == 0
    end
    for rest in (-200.0, -70.0, -65.0, 100.0)
        state = SCResettingMATNeuronState()
        state.v_rest = rest; state.tau_1 = 20.0; state.tau_2 = 250.0; state.dt = 0.25
        before = state_bits(state)
        @test reset!(state) === nothing
        @test (state.v, state.theta1, state.theta2) == (rest, 0.0, 0.0)
        @test state_bits(state)[4:end] == before[4:end]
    end
end

@testset "SC resetting-MAT complete batch refusal and caller-state recovery" begin
    for currents in ([50.0, NaN], [50.0, Inf], [50.0, -Inf], [50.0, 1.0e308])
        state = SCResettingMATNeuronState()
        before = state_bits(state)
        @test_throws ArgumentError simulate(currents; state=state)
        @test state_bits(state) == before
        result = simulate([50.0, 50.0]; state=state)
        reference = simulate([50.0, 50.0])
        @test result.state === state
        @test result.voltages == reference.voltages
        @test result.theta1 == reference.theta1 && result.theta2 == reference.theta2
        @test result.events == reference.events
    end
end

@testset "SC resetting-MAT constant-current admission and complete traces" begin
    for current in (NaN, Inf, -Inf), count in (0, 1)
        @test_throws ArgumentError simulate(count; I_ext=current)
    end
    for dt in (NaN, Inf, -Inf, 0.0, -1.0), count in (0, 1)
        @test_throws ArgumentError simulate(count; dt=dt)
    end
    @test simulate(0) == (Float64[], 0)
    @test_throws ArgumentError simulate(-1)
    for dt in (0.25, 1.0, 2.0), current in (0.0, 20.0, 50.0)
        trace, events = simulate(256; I_ext=current, dt=dt)
        state = SCResettingMATNeuronState()
        state.dt = dt
        reference = simulate(fill(current, 256); state=state)
        @test length(trace) == 256
        @test trace == reference.voltages
        @test events == sum(reference.events)
    end
    currents = vcat(zeros(32), fill(50.0, 96), repeat([20.0, 60.0], 64))
    result = simulate(currents)
    @test sum(result.events) == 13
    @test (result.state.v, result.state.theta1, result.state.theta2) == (-70.0, 5.262135955944077, 21.149478444493045)
end
end
