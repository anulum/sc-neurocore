# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia unit tests for evo_runner

using Test

include("evo_runner.jl")

using .EvoRunner: XorShift64, XorShift64_seed, next_u64!, next_f64!,
    genome_default, compute_id!, to_vector, from_vector,
    fitness_defaults, evaluate_fitness,
    FormalSafetyGuard, safety_defaults, check_safety!,
    evolve_run

@testset "XorShift64 PRNG" begin
    r = XorShift64_seed(UInt64(7))
    # Compare to Python reference values (see benchmarks/bench_evo_substrate_multilang.py).
    v1 = next_u64!(r)
    v2 = next_u64!(r)
    @test v1 == UInt64(7575888327)
    @test v2 == UInt64(8070950887952051652)
    f = next_f64!(XorShift64_seed(UInt64(7)))
    @test 0.0 <= f < 1.0
    # Seed 0 must bump to the DEAD_BEEF fallback, not stay zero.
    r0 = XorShift64_seed(UInt64(0))
    @test r0.state == UInt64(0xDEAD_BEEF_CAFE_BABE)
end

@testset "Genome to_vector / from_vector roundtrip" begin
    g = genome_default()
    g.topology.num_neurons = Int32(64)
    g.neuron.tau_fast = 7.5
    g.plasticity.stdp_lr = 0.02
    v = to_vector(g)
    back = from_vector(v, g.generation)
    @test back.topology.num_neurons == Int32(64)
    @test abs(back.neuron.tau_fast - 7.5) < 1e-12
    @test abs(back.plasticity.stdp_lr - 0.02) < 1e-12
end

@testset "Genome id is 12 hex chars and deterministic" begin
    g1 = genome_default()
    g2 = genome_default()
    compute_id!(g1)
    compute_id!(g2)
    @test g1.genome_id == g2.genome_id
    @test length(g1.genome_id) == 12
end

@testset "Fitness components match the Rust reference formula" begin
    g = genome_default()
    g.topology.num_neurons = Int32(32)
    g.topology.num_layers = Int32(2)
    g.topology.bitstream_length = Int32(256)
    spec = fitness_defaults()
    f = evaluate_fitness(spec, g)
    # accuracy = 0.5 + 0.01·32/32 = 0.51
    @test abs(f.accuracy - 0.51) < 1e-9
    # energy = 1 − 0.5·32/1024 − 0.5·256/1024 = 0.859375
    @test abs(f.energy_score - 0.859375) < 1e-9
    # latency = 1 − 2/10 = 0.8
    @test abs(f.latency_score - 0.8) < 1e-9
end

@testset "FormalSafetyGuard rejects oversized genomes" begin
    guard = FormalSafetyGuard(safety_defaults(), 0, 0)
    g = genome_default()
    g.topology.num_neurons = Int32(4096)
    @test !check_safety!(guard, g)
    @test guard.rejected == 1
end

@testset "evolve_run is deterministic under fixed seed" begin
    cfg = Dict(
        "seed" => 11, "pop_size" => 8, "n_generations" => 5,
        "elitism" => 1, "survival_fraction" => 0.5, "tournament_size" => 3,
        "crossover_prob" => 0.3, "max_age" => 20, "hall_of_fame_size" => 5,
        "stagnation_gens" => 10, "extinction_kill_fraction" => 0.9,
        "mutation" => Dict(
            "point_rate" => 0.2, "point_sigma" => 0.05,
            "structural_rate" => 0.05, "duplication_rate" => 0.01,
            "swap_rate" => 0.02, "max_neurons" => 1024, "min_neurons" => 4,
        ),
        "fitness" => Dict(
            "accuracy_bias" => 0.5, "accuracy_neuron_coef" => 0.01,
            "w_accuracy" => 0.5, "w_energy" => 0.3, "w_latency" => 0.2,
        ),
        "safety_bounds" => Dict(
            "max_neurons" => 1024, "min_neurons" => 4, "max_layers" => 16,
            "max_bitstream" => 4096, "min_bitstream" => 32, "max_connectivity" => 1.0,
        ),
        "industrial_mode" => true,
    )
    a = evolve_run(cfg)
    b = evolve_run(cfg)
    @test a["total_replications"] == b["total_replications"]
    @test length(a["final_population"]) == length(b["final_population"])
    a_ids = [g["genome_id"] for g in a["final_population"]]
    b_ids = [g["genome_id"] for g in b["final_population"]]
    @test a_ids == b_ids
end

println("Julia evo_runner unit tests: all passed")
