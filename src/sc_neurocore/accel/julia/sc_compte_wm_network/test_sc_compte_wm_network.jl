# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

using Test

include(joinpath(@__DIR__, "SCCompteWMNetwork.jl"))
using .SCCompteWMNetwork
include(joinpath(@__DIR__, "..", "neurons", "compte_wm.jl"))

const ZERO_EXC = zeros(UInt64, N_EXCITATORY)
const ZERO_INH = zeros(UInt64, N_INHIBITORY)
const ZERO_CURRENT = zeros(N_EXCITATORY)

@testset "SC Compte Julia network" begin
    @testset "fixed populations and counter fixture" begin
        runtime = SCCompteWMNetworkRuntime()
        @test length(runtime.state.v_exc_mv) == N_EXCITATORY
        @test length(runtime.state.v_inh_mv) == N_INHIBITORY
        @test runtime.spec.allow_recurrent_autapses == false
        counts = counter_poisson_counts(64, 1800.0, DT_MS, UInt64(42), UInt64(0), UInt64(0))
        @test findall(!iszero, counts) == [50, 62]
        @test sum(counts) == 2
    end

    @testset "preserved scalar-cell isolation" begin
        runtime = SCCompteWMNetworkRuntime()
        events = copy(ZERO_EXC)
        events[18] = 1
        receipt = step_with_events!(runtime, ZERO_CURRENT, events, ZERO_INH)
        original = CompteWmAccel.CompteWMNeuronState()
        @test CompteWmAccel.step!(original; external_event=true) == 0
        @test runtime.state.v_exc_mv[18] ≈ original.v atol=2e-14 rtol=0.0
        @test runtime.state.external_ampa_exc[18] ≈ original.s_ampa atol=2e-14 rtol=0.0
        @test receipt.excitatory_input_events == 1
        @test !any(receipt.excitatory_spikes)
    end

    @testset "recurrent FFT parity anchor" begin
        state = SCCompteWMNetworkState()
        state.v_exc_mv .= -60.0
        state.recurrent_nmda[[1, 38, 1025, 1902]] .= [0.2, 0.4, 0.1, 0.3]
        runtime = SCCompteWMNetworkRuntime(state=state)
        step_with_events!(runtime, ZERO_CURRENT, ZERO_EXC, ZERO_INH)
        @test runtime.state.v_exc_mv[114] ≈ -60.0099068230443 atol=2e-13 rtol=0.0
        @test runtime.state.recurrent_nmda[38] == 0.39992000800000005
    end

    @testset "deterministic receipts and seed separation" begin
        first = run!(SCCompteWMNetworkRuntime(), 0.1; statistics_window_ms=0.1)
        second = run!(SCCompteWMNetworkRuntime(), 0.1; statistics_window_ms=0.1)
        third = run!(SCCompteWMNetworkRuntime(SCCompteWMNetworkSpec(seed=UInt64(43))),
                     0.1; statistics_window_ms=0.1)
        @test first.input_sha256 == second.input_sha256
        @test first.spike_sha256 == second.spike_sha256
        @test first.final_state_sha256 == second.final_state_sha256
        @test first.input_sha256 != third.input_sha256
        @test first.final_state_sha256 != third.final_state_sha256
    end

    @testset "full-population stimulus and refractory" begin
        stimulus = SCCompteWMStimulus(0.0, 0.02, 600_000.0;
                                     kind=:global_current, center_deg=nothing)
        runtime = SCCompteWMNetworkRuntime()
        receipt = run!(runtime, 0.02; stimuli=[stimulus], statistics_window_ms=0.02)
        @test receipt.excitatory_spikes == N_EXCITATORY
        @test receipt.windows[1].statistics !== nothing
        @test all(==(-60.0), runtime.state.v_exc_mv)
        @test all(==(2.0), runtime.state.refractory_exc_ms)
        step_with_events!(runtime, ZERO_CURRENT, ZERO_EXC, ZERO_INH)
        @test all(==(-60.0), runtime.state.v_exc_mv)
    end

    @testset "atomic validation" begin
        runtime = SCCompteWMNetworkRuntime()
        before = state_sha256(runtime.state)
        invalid = copy(ZERO_CURRENT)
        invalid[5] = NaN
        @test_throws ArgumentError step_with_events!(runtime, invalid, ZERO_EXC, ZERO_INH)
        @test state_sha256(runtime.state) == before
        @test_throws ArgumentError step_with_events!(runtime, ZERO_CURRENT, ZERO_EXC[1:end-1], ZERO_INH)
        @test state_sha256(runtime.state) == before
    end

    @testset "native documentation surface" begin
        for name in (:SCCompteWMNetworkSpec, :SCCompteWMNetworkState,
                     :SCCompteWMNetworkRuntime, :SCCompteWMStepReceipt,
                     :SCCompteWMRunReceipt, :SCCompteWMStimulus,
                     :counter_poisson_counts, :step!, :step_with_events!, :run!, :reset!)
            binding = Base.Docs.Binding(SCCompteWMNetwork, name)
            @test haskey(Base.Docs.meta(SCCompteWMNetwork), binding)
        end
    end
end
