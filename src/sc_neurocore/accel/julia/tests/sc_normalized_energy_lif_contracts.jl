# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained normalized energy-LIF Julia configuration contracts

using Test
include(joinpath(@__DIR__, "../neurons/sc_normalized_energy_lif.jl"))
using .SCNormalizedEnergyLifAccel

@testset "SC normalized energy-LIF state and empty-batch validity" begin
    for field in fieldnames(SCNormalizedEnergyLIFNeuronState), bad in (NaN, Inf, -Inf)
        state = SCNormalizedEnergyLIFNeuronState()
        setfield!(state, field, bad)
        before = (state.v, state.epsilon)
        @test !valid(state)
        @test step!(state, 30.0) == -1
        @test isequal((state.v, state.epsilon), before)
        @test_throws ArgumentError simulate(Float64[]; state=state)
    end
    for rest in (-201.0, 101.0)
        state = SCNormalizedEnergyLIFNeuronState()
        state.v_rest, state.v_threshold = rest, 102.0
        @test !valid(state)
        @test_throws ArgumentError simulate(Float64[]; state=state)
        state.v_rest = -70.0
        @test valid(state)
        result = simulate(Float64[]; state=state)
        @test (result.state.v, result.state.epsilon) == (-70.0, 1.0)
    end
    state = SCNormalizedEnergyLIFNeuronState()
    result = simulate(repeat([30.0, 0.0, 50.0, 10.0], 64); state=state)
    @test sum(result.events) == 3
    @test result.state.v ≈ -52.508269792668216 atol=1e-12 rtol=0
end
