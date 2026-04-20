# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia industrial whole-process evolve runner
#
# Port of `crates/evo_substrate_core/src/runner.rs` to Julia. The wire
# contract is identical: reads an `EvolveConfig` JSON on stdin, writes
# an `EvolveResult` JSON on stdout.
#
# Industrial guards implemented: TournamentSelector, AgeRegulator,
# FormalSafetyGuard, BloatPenalizer, ExtinctionDetector, HallOfFame,
# ParetoFront, LineageTracker, MutationEngine (4 variants),
# CrossoverEngine, parametric FitnessEvaluator.

module EvoRunner

using JSON
using Random
using SHA

const GENOME_DIM = 19
const EPSILON = 1e-10

# ─── Genome + gene blocks ─────────────────────────────────────────

mutable struct TopologyGene
    num_neurons::Int32
    num_layers::Int32
    connectivity::Float64
    recurrent_fraction::Float64
    bitstream_length::Int32
end

topology_default() = TopologyGene(16, 2, 0.3, 0.1, 256)

mutable struct NeuronGene
    tau_fast::Float64
    tau_work::Float64
    tau_deep::Float64
    theta::Float64
    gamma::Float64
    delta_conf::Float64
    kappa::Float64
    w_inh::Float64
end

neuron_default() = NeuronGene(5.0, 200.0, 10000.0, 1.0, 0.2, 0.3, 5.0, 0.3)

mutable struct PlasticityGene
    stdp_lr::Float64
    stdp_tau_plus::Float64
    stdp_tau_minus::Float64
    stp_u_base::Float64
    homeostatic_rate::Float64
    meta_sensitivity::Float64
end

plasticity_default() = PlasticityGene(0.01, 20.0, 20.0, 0.5, 0.001, 1.0)

mutable struct Genome
    genome_id::String
    parent_id::String
    generation::Int32
    topology::TopologyGene
    neuron::NeuronGene
    plasticity::PlasticityGene
    weight_seed::UInt64
    identity_deep::Float64
end

genome_default() = Genome(
    "", "", 0,
    topology_default(), neuron_default(), plasticity_default(),
    UInt64(42), 0.0,
)

function to_vector(g::Genome)
    [
        Float64(g.topology.num_neurons),
        Float64(g.topology.num_layers),
        g.topology.connectivity,
        g.topology.recurrent_fraction,
        Float64(g.topology.bitstream_length),
        g.neuron.tau_fast, g.neuron.tau_work, g.neuron.tau_deep,
        g.neuron.theta, g.neuron.gamma, g.neuron.delta_conf,
        g.neuron.kappa, g.neuron.w_inh,
        g.plasticity.stdp_lr, g.plasticity.stdp_tau_plus,
        g.plasticity.stdp_tau_minus, g.plasticity.stp_u_base,
        g.plasticity.homeostatic_rate, g.plasticity.meta_sensitivity,
    ]
end

function from_vector!(g::Genome, v::Vector{Float64}, generation::Int32)
    g.generation = generation
    g.topology = TopologyGene(
        max(Int32(2), Int32(floor(v[1]))),
        max(Int32(1), Int32(floor(v[2]))),
        clamp(v[3], 0.01, 1.0),
        clamp(v[4], 0.0, 0.5),
        max(Int32(32), Int32(floor(v[5]))),
    )
    g.neuron = NeuronGene(
        max(0.5, v[6]), max(1.0, v[7]), max(10.0, v[8]),
        max(0.1, v[9]), clamp(v[10], 0.0, 1.0), clamp(v[11], 0.0, 1.0),
        max(0.1, v[12]), clamp(v[13], 0.0, 1.0),
    )
    g.plasticity = PlasticityGene(
        max(1e-6, v[14]), max(1.0, v[15]), max(1.0, v[16]),
        clamp(v[17], 0.01, 0.99), max(1e-6, v[18]), max(0.1, v[19]),
    )
    g
end

function from_vector(v::Vector{Float64}, generation::Int32)
    g = genome_default()
    from_vector!(g, v, generation)
end

"""Compute the 12-hex-char SHA-256 fingerprint; little-endian float64 bytes."""
function compute_id!(g::Genome)
    v = to_vector(g)
    bytes = reinterpret(UInt8, v)  # little-endian on x86_64
    digest = sha256(bytes)
    g.genome_id = bytes2hex(digest[1:6])
    g.genome_id
end

# ─── Mutation ─────────────────────────────────────────────────────

struct MutationConfig
    point_rate::Float64
    point_sigma::Float64
    structural_rate::Float64
    duplication_rate::Float64
    swap_rate::Float64
    max_neurons::Int32
    min_neurons::Int32
end

mutation_defaults() = MutationConfig(0.2, 0.05, 0.05, 0.01, 0.02, Int32(1024), Int32(4))

function apply_point!(cfg::MutationConfig, g::Genome, rng::AbstractRNG)
    v = to_vector(g)
    for i in 1:GENOME_DIM
        if rand(rng) < cfg.point_rate
            noise = randn(rng) * cfg.point_sigma
            v[i] += noise * (abs(v[i]) + 1e-8)
        end
    end
    from_vector!(g, v, g.generation)
end

function apply_structural!(cfg::MutationConfig, g::Genome, rng::AbstractRNG)
    delta = [-2, -1, 1, 2][rand(rng, 1:4)]
    g.topology.num_neurons = clamp(
        g.topology.num_neurons + Int32(delta), cfg.min_neurons, cfg.max_neurons,
    )
    g.topology.connectivity = clamp(g.topology.connectivity + randn(rng) * 0.05, 0.01, 1.0)
end

function apply_duplication!(cfg::MutationConfig, g::Genome)
    g.topology.num_layers = min(Int32(10), g.topology.num_layers + Int32(1))
    g.topology.num_neurons = min(cfg.max_neurons, Int32(floor(g.topology.num_neurons * 1.5)))
end

function apply_swap!(g::Genome)
    g.neuron.tau_fast, g.neuron.tau_work = g.neuron.tau_work, g.neuron.tau_fast
end

mutable struct MutationEngine
    config::MutationConfig
    rng::MersenneTwister
end

function mutate!(eng::MutationEngine, parent::Genome)
    child = deepcopy(parent)
    child.parent_id = parent.genome_id
    child.generation = parent.generation + Int32(1)
    child.identity_deep = 0.0

    roll = rand(eng.rng)
    cumulative = 0.0

    cumulative += eng.config.structural_rate
    if roll < cumulative
        apply_structural!(eng.config, child, eng.rng)
        compute_id!(child)
        return child, "structural"
    end
    cumulative += eng.config.duplication_rate
    if roll < cumulative
        apply_duplication!(eng.config, child)
        compute_id!(child)
        return child, "duplication"
    end
    cumulative += eng.config.swap_rate
    if roll < cumulative
        apply_swap!(child)
        compute_id!(child)
        return child, "swap"
    end
    apply_point!(eng.config, child, eng.rng)
    compute_id!(child)
    return child, "point"
end

# ─── Crossover ────────────────────────────────────────────────────

mutable struct CrossoverEngine
    rng::MersenneTwister
end

function crossover(eng::CrossoverEngine, a::Genome, b::Genome)
    va = to_vector(a)
    vb = to_vector(b)
    child_v = [rand(eng.rng) < 0.5 ? va[i] : vb[i] for i in 1:GENOME_DIM]
    new_gen = max(a.generation, b.generation) + Int32(1)
    child = from_vector(child_v, new_gen)
    child.parent_id = "$(a.genome_id)x$(b.genome_id)"
    compute_id!(child)
    child
end

# ─── Fitness (parametric) ─────────────────────────────────────────

struct FitnessSpec
    accuracy_bias::Float64
    accuracy_neuron_coef::Float64
    w_accuracy::Float64
    w_energy::Float64
    w_latency::Float64
end

fitness_defaults() = FitnessSpec(0.5, 0.01, 0.5, 0.3, 0.2)

mutable struct FitnessResult
    genome_id::String
    accuracy::Float64
    energy_score::Float64
    latency_score::Float64
    composite::Float64
end

function evaluate_fitness(spec::FitnessSpec, g::Genome)
    n = Float64(g.topology.num_neurons)
    layers = Float64(g.topology.num_layers)
    bitstream = Float64(g.topology.bitstream_length)

    accuracy = spec.accuracy_bias + spec.accuracy_neuron_coef * n / 32.0
    energy = max(0.0, 1.0 - 0.5 * n / 1024.0 - 0.5 * bitstream / 1024.0)
    latency = max(0.0, 1.0 - layers / 10.0)
    composite = spec.w_accuracy * accuracy + spec.w_energy * energy + spec.w_latency * latency
    FitnessResult(g.genome_id, accuracy, energy, latency, composite)
end

# ─── Guards ───────────────────────────────────────────────────────

struct SafetyBounds
    max_neurons::Int32
    min_neurons::Int32
    max_layers::Int32
    max_bitstream::Int32
    min_bitstream::Int32
    max_connectivity::Float64
end

safety_defaults() = SafetyBounds(Int32(1024), Int32(4), Int32(16),
                                 Int32(4096), Int32(32), 1.0)

mutable struct FormalSafetyGuard
    bounds::SafetyBounds
    checked::Int64
    rejected::Int64
end

function check_safety!(guard::FormalSafetyGuard, g::Genome)
    guard.checked += 1
    n_ok = g.topology.num_neurons <= guard.bounds.max_neurons
    c_ok = g.topology.connectivity <= guard.bounds.max_connectivity
    b_ok = g.topology.bitstream_length <= guard.bounds.max_bitstream
    passed = n_ok && c_ok && b_ok
    passed || (guard.rejected += 1)
    passed
end

struct BloatPenalizer
    penalty_weight::Float64
    threshold::Float64
    baseline_neurons::Int32
end

bloat_defaults() = BloatPenalizer(0.1, 2.0, Int32(16))

function bloat_score(bp::BloatPenalizer, g::Genome)
    n = Float64(g.topology.num_neurons)
    l = Float64(g.topology.num_layers)
    conn = Int64(floor(n * n * g.topology.connectivity))
    total = Int64(floor(n * 8 + l)) + conn
    base_n = Float64(bp.baseline_neurons)
    baseline = Int64(floor(base_n * 8 + 2)) + Int64(floor(base_n * base_n * 0.3))
    total / max(1, baseline)
end

function penalize(bp::BloatPenalizer, fitness::Float64, g::Genome)
    score = bloat_score(bp, g)
    if score > bp.threshold
        excess = score - bp.threshold
        return max(0.0, fitness - bp.penalty_weight * excess)
    end
    fitness
end

struct AgeRegulator
    max_age::Int32
end

# ─── Organism + population helpers ───────────────────────────────

mutable struct Organism
    genome::Genome
    fitness::Union{Nothing,FitnessResult}
    alive::Bool
    birth_generation::Int32
end

Organism(g::Genome, birth::Int32) = Organism(g, nothing, true, birth)

function cull_indices(age::AgeRegulator, pop::Vector{Organism}, current_gen::Int32)
    [i for (i, org) in enumerate(pop)
     if current_gen - org.birth_generation > age.max_age && org.alive]
end

mutable struct ExtinctionDetector
    stagnation_gens::Int
    kill_fraction::Float64
    best_history::Vector{Float64}
    extinction_count::Int64
end

extinction_new(s::Int, k::Float64) = ExtinctionDetector(s, k, Float64[], 0)

function check_extinction!(det::ExtinctionDetector, best::Float64)
    push!(det.best_history, best)
    length(det.best_history) < det.stagnation_gens && return false
    recent = det.best_history[end-det.stagnation_gens+1:end]
    if (maximum(recent) - minimum(recent)) < 1e-6
        det.extinction_count += 1
        return true
    end
    false
end

function apply_extinction!(det::ExtinctionDetector, pop::Vector{Organism},
                           rng::AbstractRNG)
    n_kill = min(Int(floor(length(pop) * det.kill_fraction)), length(pop))
    indices = collect(1:length(pop))
    shuffle!(rng, indices)
    killed = 0
    for idx in indices[1:n_kill]
        if pop[idx].alive
            pop[idx].alive = false
            killed += 1
        end
    end
    killed
end

mutable struct HallOfFame
    max_size::Int
    entries::Vector{Tuple{Float64,Genome}}
end

hall_new(n::Int) = HallOfFame(n, Tuple{Float64,Genome}[])

function update_hof!(hof::HallOfFame, org::Organism)
    org.fitness === nothing && return false
    push!(hof.entries, (org.fitness.composite, deepcopy(org.genome)))
    sort!(hof.entries, by=x -> -x[1])
    length(hof.entries) > hof.max_size && (hof.entries = hof.entries[1:hof.max_size])
    true
end

mutable struct ParetoFront
    front::Vector{Organism}
end

pareto_new() = ParetoFront(Organism[])

function dominates(a::FitnessResult, b::FitnessResult)
    va = (a.accuracy, a.energy_score, a.latency_score)
    vb = (b.accuracy, b.energy_score, b.latency_score)
    at_least_one_better = false
    for (x, y) in zip(va, vb)
        x < y && return false
        x > y && (at_least_one_better = true)
    end
    at_least_one_better
end

function update_pareto!(pf::ParetoFront, org::Organism)
    org.fitness === nothing && return false
    for existing in pf.front
        existing.fitness === nothing && continue
        dominates(existing.fitness, org.fitness) && return false
    end
    pf.front = [o for o in pf.front
                if o.fitness === nothing || !dominates(org.fitness, o.fitness)]
    push!(pf.front, deepcopy(org))
    true
end

struct TournamentSelector
    tournament_size::Int
end

function select_tournament(ts::TournamentSelector, pop::Vector{Organism},
                           rng::AbstractRNG)
    isempty(pop) && return nothing
    k = min(ts.tournament_size, length(pop))
    seen = Int[]
    best = nothing
    best_fit = -Inf
    while length(seen) < k
        idx = rand(rng, 1:length(pop))
        idx in seen && continue
        push!(seen, idx)
        org = pop[idx]
        fit = org.fitness === nothing ? 0.0 : org.fitness.composite
        if fit > best_fit
            best_fit = fit
            best = org
        end
    end
    best
end

# ─── Lineage ──────────────────────────────────────────────────────

struct LineageRecord
    genome_id::String
    parent_id::String
    generation::Int32
    mutation_type::String
    fitness::Float64
end

mutable struct LineageTracker
    records::Vector{LineageRecord}
end

lineage_new() = LineageTracker(LineageRecord[])

function record_lineage!(lt::LineageTracker, org::Organism, mtype::String)
    fit = org.fitness === nothing ? 0.0 : org.fitness.composite
    push!(lt.records, LineageRecord(org.genome.genome_id, org.genome.parent_id,
                                    org.genome.generation, mtype, fit))
end

# ─── Diversity ────────────────────────────────────────────────────

function pairwise_diversity(pop::Vector{Organism})
    alive = [o for o in pop if o.alive]
    length(alive) < 2 && return 0.0
    acc = 0.0
    count = 0.0
    for i in 1:length(alive)
        va = to_vector(alive[i].genome)
        for j in (i+1):length(alive)
            vb = to_vector(alive[j].genome)
            s = 0.0
            for k in 1:GENOME_DIM
                s += abs(va[k] - vb[k]) / (abs(va[k]) + abs(vb[k]) + EPSILON)
            end
            acc += s / GENOME_DIM
            count += 1.0
        end
    end
    acc / count
end

# ─── Main runner ──────────────────────────────────────────────────

function evolve_run(cfg::AbstractDict)
    master_rng = MersenneTwister(UInt32(cfg["seed"]))

    mutation_cfg = MutationConfig(
        cfg["mutation"]["point_rate"], cfg["mutation"]["point_sigma"],
        cfg["mutation"]["structural_rate"], cfg["mutation"]["duplication_rate"],
        cfg["mutation"]["swap_rate"],
        Int32(cfg["mutation"]["max_neurons"]), Int32(cfg["mutation"]["min_neurons"]),
    )
    fitness_spec = FitnessSpec(
        cfg["fitness"]["accuracy_bias"], cfg["fitness"]["accuracy_neuron_coef"],
        cfg["fitness"]["w_accuracy"], cfg["fitness"]["w_energy"], cfg["fitness"]["w_latency"],
    )
    safety = SafetyBounds(
        Int32(cfg["safety_bounds"]["max_neurons"]),
        Int32(cfg["safety_bounds"]["min_neurons"]),
        Int32(cfg["safety_bounds"]["max_layers"]),
        Int32(cfg["safety_bounds"]["max_bitstream"]),
        Int32(cfg["safety_bounds"]["min_bitstream"]),
        cfg["safety_bounds"]["max_connectivity"],
    )

    mutator = MutationEngine(mutation_cfg,
                             MersenneTwister(UInt32(rand(master_rng, UInt32))))
    xover = CrossoverEngine(MersenneTwister(UInt32(rand(master_rng, UInt32))))
    guard = FormalSafetyGuard(safety, 0, 0)
    bloat = bloat_defaults()
    age = AgeRegulator(Int32(cfg["max_age"]))
    extinction = extinction_new(cfg["stagnation_gens"], cfg["extinction_kill_fraction"])
    hof = hall_new(cfg["hall_of_fame_size"])
    pareto = pareto_new()
    tournament = TournamentSelector(cfg["tournament_size"])
    lineage = lineage_new()

    pop_size = cfg["pop_size"]
    n_generations = cfg["n_generations"]
    industrial = cfg["industrial_mode"]
    elitism = cfg["elitism"]
    survival_fraction = cfg["survival_fraction"]
    crossover_prob = cfg["crossover_prob"]

    # Seed population
    population = Organism[]
    for _ in 1:pop_size
        g = genome_default()
        compute_id!(g)
        org = Organism(g, Int32(0))
        record_lineage!(lineage, org, "seed")
        push!(population, org)
    end

    total_replications = Int64(0)
    stats = Vector{Dict{String,Any}}()

    for gen in 1:n_generations
        gen_i32 = Int32(gen)

        # 1. Evaluate + update HoF / Pareto
        for org in population
            org.alive || continue
            fit = evaluate_fitness(fitness_spec, org.genome)
            if industrial
                fit.composite = penalize(bloat, fit.composite, org.genome)
            end
            org.fitness = fit
            update_hof!(hof, org)
            update_pareto!(pareto, org)
        end

        # 2. Industrial culls
        killed = 0
        if industrial
            for idx in cull_indices(age, population, gen_i32)
                population[idx].alive = false
                killed += 1
            end
            best = 0.0
            for o in population
                o.alive || continue
                o.fitness === nothing && continue
                best = max(best, o.fitness.composite)
            end
            if check_extinction!(extinction, best)
                killed += apply_extinction!(extinction, population, mutator.rng)
            end
        end

        # 3. Survival cull
        alive_sorted = [i for (i, o) in enumerate(population)
                        if o.alive && o.fitness !== nothing]
        sort!(alive_sorted, by=i -> -population[i].fitness.composite)
        keep = max(elitism + 1,
                   Int(floor(length(alive_sorted) * survival_fraction)))
        for idx in alive_sorted[keep+1:end]
            population[idx].alive = false
            killed += 1
        end
        population = [o for o in population if o.alive]

        # 4. Replicate
        survivors = deepcopy(population)
        children = 0
        while length(population) < pop_size && !isempty(survivors)
            parent = industrial ? select_tournament(tournament, survivors, mutator.rng) :
                                  survivors[1]
            parent === nothing && break
            partner = industrial ? select_tournament(tournament, survivors, mutator.rng) :
                                   (length(survivors) > 1 ? survivors[2] : nothing)

            local child_genome
            if partner !== nothing && rand(mutator.rng) < crossover_prob
                c = crossover(xover, parent.genome, partner.genome)
                c.generation = gen_i32
                child_genome = c
            else
                (c, _mt) = mutate!(mutator, parent.genome)
                c.generation = gen_i32
                child_genome = c
            end

            check_safety!(guard, child_genome) || continue

            total_replications += 1
            child = Organism(child_genome, gen_i32)
            record_lineage!(lineage, child, "replicate")
            push!(population, child)
            children += 1
        end

        # 5. Stats
        best_fitness = 0.0
        fits = Float64[]
        for o in population
            o.fitness === nothing && continue
            push!(fits, o.fitness.composite)
            best_fitness = max(best_fitness, o.fitness.composite)
        end
        mean_fitness = isempty(fits) ? 0.0 : sum(fits) / length(fits)

        push!(stats, Dict(
            "generation" => gen,
            "population_size" => length(population),
            "best_fitness" => best_fitness,
            "mean_fitness" => mean_fitness,
            "diversity" => pairwise_diversity(population),
            "killed" => killed,
            "children" => children,
            "extinctions" => extinction.extinction_count,
            "safety_rejections" => guard.rejected,
        ))
    end

    # Serialise genome records to JSON-safe dicts
    genome_to_dict = g -> Dict(
        "genome_id" => g.genome_id,
        "parent_id" => g.parent_id,
        "generation" => g.generation,
        "num_neurons" => g.topology.num_neurons,
        "num_layers" => g.topology.num_layers,
        "connectivity" => g.topology.connectivity,
        "bitstream_length" => g.topology.bitstream_length,
        "tau_fast" => g.neuron.tau_fast,
        "tau_work" => g.neuron.tau_work,
        "tau_deep" => g.neuron.tau_deep,
    )

    Dict(
        "final_population" => [genome_to_dict(o.genome) for o in population],
        "stats_per_generation" => stats,
        "hall_of_fame" => [genome_to_dict(g) for (_, g) in hof.entries],
        "pareto_front" => [genome_to_dict(o.genome) for o in pareto.front],
        "lineage" => [Dict("genome_id" => r.genome_id, "parent_id" => r.parent_id,
                           "generation" => r.generation, "mutation_type" => r.mutation_type,
                           "fitness" => r.fitness) for r in lineage.records],
        "total_replications" => total_replications,
        "safety_checked" => guard.checked,
        "safety_rejected" => guard.rejected,
        "extinction_count" => extinction.extinction_count,
    )
end

end  # module

if abspath(PROGRAM_FILE) == @__FILE__
    import JSON
    cfg = JSON.parse(read(stdin, String))
    result = EvoRunner.evolve_run(cfg)
    JSON.print(stdout, result)
end
