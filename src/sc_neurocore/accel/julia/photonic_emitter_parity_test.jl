# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia photonic crosstalk parity tests

using Test

include(joinpath(@__DIR__, "optics", "photonic_emitter.jl"))
using .PhotonicEmitterAccel

@testset "photonic crosstalk parity" begin
    metrics = analyze_pair(200.0, 50.0)
    @test metrics.coupling_coefficient_per_um ≈ 0.015593714868342372 atol = 1.0e-15 rtol = 0.0
    @test metrics.coupling_ratio ≈ 0.49428770428966934 atol = 1.0e-15 rtol = 0.0
    @test metrics.isolation_db ≈ 3.0602019274692 atol = 1.0e-15 rtol = 0.0
    @test analyze_pair(200.0, 0.0).isolation_db == 300.0

    single = analyze_bank(1, 200.0, 50.0)
    @test single.num_near_pairs == 0
    @test single.num_far_pairs == 0
    @test isinf(single.worst_isolation_db)

    pairs = analyze_pairs([PairSpec(0, 1, 200.0, 50.0), PairSpec(1, 2, 400.0, 50.0)])
    @test length(pairs) == 2
    @test pairs[1].coupling_ratio >= pairs[2].coupling_ratio

    @test_throws DomainError analyze_pair(NaN, 10.0)
    @test_throws DomainError analyze_pair(200.0, -1.0)
    @test_throws DomainError analyze_pair(200.0, 10.0, 0.0)
    @test_throws DomainError analyze_pair(200.0, 10.0, 1550.0, 1.45, 1.45)
    @test_throws DomainError analyze_bank(0, 200.0, 10.0)
    @test_throws DomainError analyze_pairs([PairSpec(1, 1, 200.0, 10.0)])
end
