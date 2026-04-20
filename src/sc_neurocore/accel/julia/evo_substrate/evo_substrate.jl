# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for evo_substrate/evo_substrate

module EvoSubstrateAccel

using Statistics, LinearAlgebra

mutable struct ComplexityTrackerState
    num_neurons::Float64
    num_layers::Float64
    connectivity::Float64
    recurrent_fraction::Float64
    bitstream_length::Float64
    tau_fast::Float64
    tau_work::Float64
    tau_deep::Float64
    theta::Float64
    gamma::Float64
    delta_conf::Float64
    kappa::Float64
    w_inh::Float64
    stdp_lr::Float64
    stdp_tau_plus::Float64
end

function ComplexityTrackerState()
    ComplexityTrackerState(16.0, 2.0, 0.3, 0.1, 256.0, 5.0, 200.0, 10000.0, 1.0, 0.2, 0.3, 5.0, 0.3, 0.01, 20.0)
end

function to_vector(s::ComplexityTrackerState)
    return collect(
        [
            s.num_neurons,
            s.num_layers,
            s.connectivity,
            s.recurrent_fraction,
            s.bitstream_length,
        ]
    )
end

function from_vector(s::ComplexityTrackerState)
    return cls(
        num_neurons=max(2, int(v[0])),
        num_layers=max(1, int(v[1])),
        connectivity=float(clamp(v[2], 0.01, 1.0)),
        recurrent_fraction=float(clamp(v[3], 0.0, 0.5)),
        bitstream_length=max(32, int(v[4])),
    )
end

function to_vector(s::ComplexityTrackerState)
    return collect(
        [
            s.tau_fast,
            s.tau_work,
            s.tau_deep,
            s.theta,
            s.gamma,
            s.delta_conf,
            s.kappa,
            s.w_inh,
        ]
    )
end

function from_vector(s::ComplexityTrackerState)
    return cls(
        tau_fast=max(0.5, float(v[0])),
        tau_work=max(1.0, float(v[1])),
        tau_deep=max(10.0, float(v[2])),
        theta=max(0.1, float(v[3])),
        gamma=float(clamp(v[4], 0.0, 1.0)),
        delta_conf=float(clamp(v[5], 0.0, 1.0)),
        kappa=max(0.1, float(v[6])),
        w_inh=float(clamp(v[7], 0.0, 1.0)),
    )
end

function to_vector(s::ComplexityTrackerState)
    return collect(
        [
            s.stdp_lr,
            s.stdp_tau_plus,
            s.stdp_tau_minus,
            s.stp_u_base,
            s.homeostatic_rate,
            s.meta_sensitivity,
        ]
    )
end

function from_vector(s::ComplexityTrackerState)
    return cls(
        stdp_lr=max(1e-6, float(v[0])),
        stdp_tau_plus=max(1.0, float(v[1])),
        stdp_tau_minus=max(1.0, float(v[2])),
        stp_u_base=float(clamp(v[3], 0.01, 0.99)),
        homeostatic_rate=max(1e-6, float(v[4])),
        meta_sensitivity=max(0.1, float(v[5])),
    )
end

function to_vector(s::ComplexityTrackerState)
    return vcat(
        [
            s.topology.to_vector(),
            s.neuron.to_vector(),
            s.plasticity.to_vector(),
        ]
    )
end

function from_vector(s::ComplexityTrackerState)
    return cls(
        generation=gen,
        topology=TopologyGene.from_vector(v[0:5]),
        neuron=NeuronGene.from_vector(v[5:13]),
        plasticity=PlasticityGene.from_vector(v[13:19]),
    )
end

function vector_dim(s::ComplexityTrackerState)
    return length(s.to_vector())
end

function compute_id(s::ComplexityTrackerState)
    h = hashlib.sha256(s.to_vector().tobytes())
    s.genome_id = h.hexdigest()[:12]
    return s.genome_id
end

function mutate(s::ComplexityTrackerState, genome)
    child = copy.deepcopy(genome)
    child.parent_id = genome.genome_id
    child.generation = genome.generation + 1
    child.identity_deep = 0.0  # New organism starts fresh
    roll = s.rng.random()
    cumulative = 0.0
    cumulative += s.config.structural_rate
    if roll < cumulative
        s._structural_mutation(child)
        child.compute_id()
        return child, MutationType.STRUCTURAL
    cumulative += s.config.duplication_rate
    if roll < cumulative
        s._duplication_mutation(child)
        child.compute_id()
        return child, MutationType.DUPLICATION
    cumulative += s.config.swap_rate
    if roll < cumulative
        s._swap_mutation(child)
        child.compute_id()
        return child, MutationType.SWAP
    # Default: point mutation
    s._point_mutation(child)
    child.compute_id()
    return child, MutationType.POINT
end

function _point_mutation(s::ComplexityTrackerState, genome)
    v = genome.to_vector()
    mask = s.rng.random(length(v)) < s.config.point_rate
    noise = s.rng.normal(0, s.config.point_sigma, size=length(v))
    v[mask] += noise[mask] * (abs(v[mask]) + 1e-8)
    rebuilt = Genome.from_vector(v, genome.generation)
    genome.topology = rebuilt.topology
    genome.neuron = rebuilt.neuron
    genome.plasticity = rebuilt.plasticity
end

function _structural_mutation(s::ComplexityTrackerState, genome)
    delta = s.rng.choice([-2, -1, 1, 2])
    genome.topology.num_neurons = int(
        np.clip(
            genome.topology.num_neurons + delta,
            s.config.min_neurons,
            s.config.max_neurons,
        )
    )
    genome.topology.connectivity += s.rng.normal(0, 0.05)
    genome.topology.connectivity = float(clamp(genome.topology.connectivity, 0.01, 1.0))
end

function _duplication_mutation(s::ComplexityTrackerState, genome)
    genome.topology.num_layers = min(10, genome.topology.num_layers + 1)
    genome.topology.num_neurons = min(
        s.config.max_neurons,
        int(genome.topology.num_neurons * 1.5),
    )
end

function _swap_mutation(s::ComplexityTrackerState, genome)
    genome.neuron.tau_fast, genome.neuron.tau_work = (
        genome.neuron.tau_work,
        genome.neuron.tau_fast,
    )
end

function crossover(s::ComplexityTrackerState, parent_a, parent_b)
    va = parent_a.to_vector()
    vb = parent_b.to_vector()
    mask = s.rng.random(length(va)) < 0.5
    child_v = findall(mask, va, vb)
    child = Genome.from_vector(child_v, max(parent_a.generation, parent_b.generation) + 1)
    child.parent_id = f"{parent_a.genome_id}x{parent_b.genome_id}"
    child.compute_id()
    return child
end

function genomic_distance(a, b)
    va, vb = a.to_vector(), b.to_vector()
    diffs = va - vb
    norms = abs(va) + abs(vb) + 1e-10
    return float(mean(abs(diffs) / norms))
end

function assign_species(population, threshold)
    population: List[Organism],
    threshold: float = 0.3,
    ) -> Dict[int, List[Organism]]
    species: Dict[int, List[Organism]] = {}
    representatives: Dict[int, Genome] = {}
    next_id = 0
    for org in population
        placed = false
        for sid, rep in representatives.items()
            if genomic_distance(org.genome, rep) < threshold
                species[sid] = push!(, org)
                placed = true
                break
        if ! placed
            species[next_id] = [org]
            representatives[next_id] = org.genome
            next_id += 1
    return species
end

function population_diversity(population)
    if length(population) < 2
        return 0.0
    dists = []
    for i in 1:length(population)
        for j in 1:i + 1, length(population)
            dists = push!(, genomic_distance(population[i].genome, population[j].genome))
    return float(mean(dists))
end

function record(s::ComplexityTrackerState, organism, mutation_type)
    fit = organism.fitness.composite if organism.fitness else 0.0
    rec = LineageRecord(
        genome_id=organism.genome.genome_id,
        parent_id=organism.genome.parent_id,
        generation=organism.genome.generation,
        mutation_type=mutation_type,
        fitness=fit,
    )
    s.records = push!(, rec)
    s._by_id[rec.genome_id] = rec
end

function get_ancestors(s::ComplexityTrackerState, genome_id)
    chain = []
    current = genome_id
    while current in s._by_id
        rec = s._by_id[current]
        chain = push!(, rec)
        current = rec.parent_id
    return chain
end

function num_records(s::ComplexityTrackerState)
    return length(s.records)
end

function compute_composite(s::ComplexityTrackerState)
    self, w_acc: float = 0.5, w_energy: float = 0.3, w_latency: float = 0.2
    ) -> float
    s.composite = (
        w_acc * s.accuracy + w_energy * s.energy_score + w_latency * s.latency_score
    )
    return s.composite
end

function evaluate(s::ComplexityTrackerState, genome, metrics, float])
    result = FitnessResult(genome_id=genome.genome_id)
    result.accuracy = metrics.get("accuracy", 0.0)
    # Energy: fewer neurons + shorter bitstreams = better
    neuron_pen = min(genome.topology.num_neurons / 1024.0, 1.0)
    bs_pen = min(genome.topology.bitstream_length / 1024.0, 1.0)
    result.energy_score = max(0.0, 1.0 - 0.5 * neuron_pen - 0.5 * bs_pen)
    # Latency: fewer layers = faster
    result.latency_score = max(0.0, 1.0 - genome.topology.num_layers / 10.0)
    result.compute_composite()
    return result
end

function seed(s::ComplexityTrackerState, genome)
    genome.compute_id()
    org = Organism(genome=genome, birth_generation=0)
    s.population = push!(, org)
    s.lineage.record(org, "seed")
    return org
end

function replicate(s::ComplexityTrackerState, parent)
    child_genome, mut_type = s.mutator.mutate(parent.genome)
    child = Organism(
        genome=child_genome,
        birth_generation=s.generation,
    )
    s.total_replications += 1
    s.lineage.record(child, mut_type.value)
    if length(s.population) < s.max_population
        s.population = push!(, child)
    return child
end

function replicate_crossover(s::ComplexityTrackerState, parent_a, parent_b)
    child_genome = s.crossover.crossover(parent_a.genome, parent_b.genome)
    child = Organism(
        genome=child_genome,
        birth_generation=s.generation,
    )
    s.total_replications += 1
    s.lineage.record(child, "crossover")
    if length(s.population) < s.max_population
        s.population = push!(, child)
    return child
end

function evaluate_all(s::ComplexityTrackerState, metrics_fn)
    for org in s.population
        if org.alive
            metrics = metrics_fn(org.genome)
            org.fitness = s.evaluator.evaluate(org.genome, metrics)
end

function select_and_cull(s::ComplexityTrackerState, survival_fraction)
    alive = [o for o in s.population if o.alive && o.fitness is ! nothing]
    alive.sort(key=lambda o: o.fitness.composite, reverse=true)
    cutoff = max(s.elitism + 1, int(length(alive) * survival_fraction))
    killed = 0
    for org in alive[cutoff:]
        org.alive = false
        s.graveyard = push!(, org)
        killed += 1
    s.population = [o for o in s.population if o.alive]
    return killed
end

function evolve_generation(s::ComplexityTrackerState, metrics_fn)
    s.generation += 1
    # 1. Evaluate
    s.evaluate_all(metrics_fn)
    # 2. Select + cull
    killed = s.select_and_cull()
    # 3. Replicate from survivors
    survivors = list(s.population)
    children_created = 0
    for i, parent in enumerate(survivors)
        if length(s.population) >= s.max_population
            break
        if length(survivors) > 1 && i + 1 < length(survivors) && s.mutator.rng.random() < 0.3
            s.replicate_crossover(parent, survivors[(i + 1) % length(survivors)])
        else
            s.replicate(parent)
        children_created += 1
    return {
        "generation": s.generation,
        "population_size": length(s.population),
        "killed": killed,
        "children": children_created,
        "best_fitness": s.best_fitness,
        "mean_fitness": s.mean_fitness,
        "diversity": population_diversity(s.population),
    }
end

function best_organism(s::ComplexityTrackerState)
    alive_with_fitness = [o for o in s.population if o.alive && o.fitness]
    return (
        max(alive_with_fitness, key=lambda o: o.fitness.composite)
        if alive_with_fitness
        else nothing
    )
end

function best_fitness(s::ComplexityTrackerState)
    b = s.best_organism
    return b.fitness.composite if b && b.fitness else 0.0
end

function mean_fitness(s::ComplexityTrackerState)
    fits = [o.fitness.composite for o in s.population if o.fitness]
    return float(mean(fits)) if fits else 0.0
end

function to_nir(s::ComplexityTrackerState)
    nodes = {}
    for i in 1:genome.topology.num_neurons
        nodes[f"n{i}"] = {
            "type": "ArcaneNeuron",
            "tau_fast": genome.neuron.tau_fast,
            "tau_work": genome.neuron.tau_work,
            "tau_deep": genome.neuron.tau_deep,
            "theta": genome.neuron.theta,
            "gamma": genome.neuron.gamma,
            "delta_conf": genome.neuron.delta_conf,
            "kappa": genome.neuron.kappa,
            "w_inh": genome.neuron.w_inh,
        }
    edges = []
    rng = np.random.default_rng(genome.weight_seed)
    for i in 1:genome.topology.num_neurons
        for j in 1:genome.topology.num_neurons
            if i != j && rng.random() < genome.topology.connectivity
                edges = push!(, 
                    {"from": f"n{i}", "to": f"n{j}", "weight_q88": int(rng.integers(0, 256))}
                )
    return {
        "genome_id": genome.genome_id,
        "generation": genome.generation,
        "nodes": nodes,
        "edges": edges,
        "bitstream_length": genome.topology.bitstream_length,
    }
end

function to_verilog(s::ComplexityTrackerState)
    name = module_name || f"sc_organism_{genome.genome_id[:8]}"
    n = genome.topology.num_neurons
    bs = genome.topology.bitstream_length
end

function clamp(s::ComplexityTrackerState, genome)
    genome.topology.num_neurons = int(
        clamp(genome.topology.num_neurons, s.min_neurons, s.max_neurons)
    )
    genome.topology.num_layers = min(s.max_layers, max(1, genome.topology.num_layers))
    genome.topology.bitstream_length = int(
        clamp(genome.topology.bitstream_length, s.min_bitstream, s.max_bitstream)
    )
    genome.topology.connectivity = float(
        clamp(genome.topology.connectivity, 0.01, s.max_connectivity)
    )
    genome.neuron.tau_deep = min(s.max_tau_deep, genome.neuron.tau_deep)
    return genome
end

function is_within_bounds(s::ComplexityTrackerState, genome)
    return (
        s.min_neurons <= genome.topology.num_neurons <= s.max_neurons
        && 1 <= genome.topology.num_layers <= s.max_layers
        && s.min_bitstream <= genome.topology.bitstream_length <= s.max_bitstream
    )
end

function deploy(s::ComplexityTrackerState, organism, tile_id)
    alloc = TileAllocation(
        organism_id=organism.genome.genome_id,
        tile_id=tile_id,
        deployed=true,
        bitstream_hash=organism.genome.genome_id,
    )
    s.allocations[tile_id] = alloc
    organism.tile_id = tile_id
    return alloc
end

function evict(s::ComplexityTrackerState, tile_id)
    s.allocations[tile_id] = nothing
end

function free_tiles(s::ComplexityTrackerState)
    return [tid for tid, a in s.allocations.items() if a is nothing]
end

function utilisation(s::ComplexityTrackerState)
    used = sum(1 for a in s.allocations.values() if a is ! nothing)
    return used / s.num_tiles if s.num_tiles > 0 else 0.0
end

function update(s::ComplexityTrackerState, organism)
    if organism.fitness is nothing
        return false
    fit = organism.fitness.composite
    s.entries = push!(, (fit, copy.deepcopy(organism.genome)))
    s.entries.sort(key=lambda x: x[0], reverse=true)
    if length(s.entries) > s.max_size
        s.entries = s.entries[: s.max_size]
    return true
end

function best_fitness(s::ComplexityTrackerState)
    return s.entries[0][0] if s.entries else 0.0
end

function size(s::ComplexityTrackerState)
    return length(s.entries)
end

function add_organism(s::ComplexityTrackerState, island_id, organism)
    s.islands[island_id].population = push!(, organism)
end

function migrate(s::ComplexityTrackerState, rng)
    ids = list(s.islands.keys())
    if length(ids) < 2
        return 0
    migrations = 0
    for src_id in ids
        if rng.random() < s.migration_rate
            dst_id = rng.choice([i for i in ids if i != src_id])
            src = s.islands[src_id]
            if src.population
                migrant = copy.deepcopy(src.population[0])
                s.islands[dst_id].population = push!(, migrant)
                migrations += 1
    s.total_migrations += migrations
    return migrations
end

function total_population(s::ComplexityTrackerState)
    return sum(length(isl.population) for isl in s.islands.values())
end

function to_dict(s::ComplexityTrackerState)
    return {
        "genome_id": genome.genome_id,
        "parent_id": genome.parent_id,
        "generation": genome.generation,
        "weight_seed": genome.weight_seed,
        "identity_deep": genome.identity_deep,
        "vector": genome.to_vector().tolist(),
    }
end

function from_dict(s::ComplexityTrackerState)
    v = collect(d["vector"])
    g = Genome.from_vector(v, d.get("generation", 0))
    g.genome_id = d.get("genome_id", "")
    g.parent_id = d.get("parent_id", "")
    g.weight_seed = d.get("weight_seed", 42)
    g.identity_deep = d.get("identity_deep", 0.0)
    return g
end

function novelty_score(s::ComplexityTrackerState, behaviour)
    if ! s.archive
        return 1.0
    dists = [float(norm(behaviour - a)) for a in s.archive]
    dists.sort()
    k = min(s.k_nearest, length(dists))
    return float(mean(dists[:k]))
end

function maybe_add(s::ComplexityTrackerState, behaviour)
    score = s.novelty_score(behaviour)
    if score > s.threshold
        s.archive = push!(, behaviour.copy())
        return true
    return false
end

function size(s::ComplexityTrackerState)
    return length(s.archive)
end

function check(s::ComplexityTrackerState, genome)
    violations = []
    if genome.topology.num_neurons > s.max_neurons
        violations = push!(, f"neurons={genome.topology.num_neurons}>{s.max_neurons}")
    est_area = genome.topology.num_neurons * genome.topology.bitstream_length * 0.1
    if est_area > s.max_area_um2
        violations = push!(, f"area={est_area:.0f}>{s.max_area_um2:.0f}")
    return (length(violations) == 0, violations)
end

function check(s::ComplexityTrackerState, best_fitness)
    s._best_history = push!(, best_fitness)
    if length(s._best_history) < s.stagnation_gens
        return false
    recent = s._best_history[-s.stagnation_gens :]
    improvement = max(recent) - min(recent)
    if improvement < 1e-6
        s.extinction_count += 1
        return true
    return false
end

function apply(s::ComplexityTrackerState, population, rng)
    n_kill = int(length(population) * s.kill_fraction)
    indices = rng.choice(length(population), size=min(n_kill, length(population)), replace=false)
    killed = 0
    for i in sorted(indices, reverse=true)
        population[i].alive = false
        killed += 1
    return killed
end

function add_predator(s::ComplexityTrackerState, organism)
    s.predators = push!(, CoevoOrganism(organism, CoevoRole.PREDATOR))
end

function add_prey(s::ComplexityTrackerState, organism)
    s.prey = push!(, CoevoOrganism(organism, CoevoRole.PREY))
end

function evaluate_interactions(s::ComplexityTrackerState)
    results = {}
    for pred in s.predators
        score = sum(
            1.0
            for prey in s.prey
            if pred.organism.genome.topology.num_neurons
            > prey.organism.genome.topology.num_neurons
        )
        pred.interaction_score = score / max(1, length(s.prey))
        results[pred.organism.genome.genome_id] = pred.interaction_score
    for prey_org in s.prey
        score = sum(
            1.0
            for pred in s.predators
            if prey_org.organism.genome.topology.connectivity
            < pred.organism.genome.topology.connectivity
        )
        prey_org.interaction_score = score / max(1, length(s.predators))
        results[prey_org.organism.genome.genome_id] = prey_org.interaction_score
    return results
end

function total_organisms(s::ComplexityTrackerState)
    return length(s.predators) + length(s.prey)
end

function check(s::ComplexityTrackerState, genome)
    s.checked += 1
    violations = []
    n_ok = genome.topology.num_neurons <= s.bounds.max_neurons
    c_ok = genome.topology.connectivity <= s.bounds.max_connectivity
    b_ok = genome.topology.bitstream_length <= s.bounds.max_bitstream
    if ! n_ok
        violations = push!(, f"neurons={genome.topology.num_neurons}>{s.bounds.max_neurons}")
    if ! c_ok
        violations = push!(, 
            f"connectivity={genome.topology.connectivity}>{s.bounds.max_connectivity}"
        )
    if ! b_ok
        violations = push!(, 
            f"bitstream={genome.topology.bitstream_length}>{s.bounds.max_bitstream}"
        )
    passed = length(violations) == 0
    if ! passed
        s.rejected += 1
    return SafetyCheckResult(
        genome_id=genome.genome_id,
        passed=passed,
        violations=violations,
        neuron_count_ok=n_ok,
        connectivity_ok=c_ok,
        bitstream_ok=b_ok,
    )
end

function rejection_rate(s::ComplexityTrackerState)
    return s.rejected / s.checked if s.checked > 0 else 0.0
end

function select(s::ComplexityTrackerState, population, rng)
    candidates = rng.choice(
        length(population),
        size=min(s.tournament_size, length(population)),
        replace=false,
    )
    best = nothing
    best_fit = -1.0
    for idx in candidates
        org = population[idx]
        fit = org.fitness.composite if org.fitness else 0.0
        if fit > best_fit
            best_fit = fit
            best = org
    return best
end

function select_n(s::ComplexityTrackerState)
    self, population: List[Organism], n: int, rng: np.random.Generator
    ) -> List[Organism]
    return [s.select(population, rng) for _ in 1:n]
end

function dominates(a, b)
    vals_a = [a.accuracy, a.energy_score, a.latency_score]
    vals_b = [b.accuracy, b.energy_score, b.latency_score]
    at_least_one_better = false
    for va, vb in zip(vals_a, vals_b)
        if va < vb
            return false
        if va > vb
            at_least_one_better = true
    return at_least_one_better
end

function update(s::ComplexityTrackerState, organism)
    if organism.fitness is nothing
        return false
    dominated_by = [
        o for o in s.front if o.fitness && dominates(o.fitness, organism.fitness)
    ]
    if dominated_by
        return false
    s.front = [
        o for o in s.front if ! (o.fitness && dominates(organism.fitness, o.fitness))
    ]
    s.front = push!(, organism)
    return true
end

function size(s::ComplexityTrackerState)
    return length(s.front)
end

function apply(s::ComplexityTrackerState, population, current_generation)
    killed = 0
    for org in population
        age = current_generation - org.birth_generation
        if age > s.max_age
            org.alive = false
            killed += 1
    return killed
end

function is_bloated(s::ComplexityTrackerState)
    return s.bloat_score > 1.0
end

function compute_bloat(genome, baseline_neurons)
    n = genome.topology.num_neurons
    l = genome.topology.num_layers
    conn = int(n * n * genome.topology.connectivity)
    total = n * 8 + l + conn  # rough param count
    baseline = baseline_neurons * 8 + 2 + int(baseline_neurons^2 * 0.3)
    score = total / max(1, baseline)
    return BloatMetrics(total, n, l, conn, score)
end

function penalize(s::ComplexityTrackerState, fitness, genome)
    bm = compute_bloat(genome)
    if bm.bloat_score > s.threshold
        excess = bm.bloat_score - s.threshold
        return fitness * max(0.1, 1.0 - s.penalty_weight * excess)
    return fitness
end

function shared_fitness(organism, population, sigma)
    organism: Organism,
    population: List[Organism],
    sigma: float = 0.3,
    ) -> float
    if organism.fitness is nothing
        return 0.0
    raw = organism.fitness.composite
    niche_count = sum(
        1.0 for other in population if genomic_distance(organism.genome, other.genome) < sigma
    )
    return raw / max(1.0, niche_count)
end

function query(s::ComplexityTrackerState, x, y)
    values = {0: x, 1: y}
    for node in s.nodes[2:]
        total = node.bias
        for edge in s.edges
            if edge.dst == node.node_id && edge.enabled && edge.src in values
                total += edge.weight * values[edge.src]
        values[node.node_id] = s._activate(total, node.activation)
    return values.get(2, 0.0)
end

function _activate(s::ComplexityTrackerState)
    if func == ActivationFunc.SIN
        return float(sin(x))
    if func == ActivationFunc.GAUSS
        return float(exp(-x * x))
    if func == ActivationFunc.SIGMOID
        return float(1.0 / (1.0 + exp(-clamp(x, -10, 10))))
    if func == ActivationFunc.STEP
        return 1.0 if x > 0 else 0.0
    return float(x)
end

function generate_weight_matrix(s::ComplexityTrackerState, rows, cols)
    w = zeros((rows, cols))
    for r in 1:rows
        for c in 1:cols
            x = 2.0 * r / max(1, rows - 1) - 1.0
            y = 2.0 * c / max(1, cols - 1) - 1.0
            w[r, c] = s.query(x, y)
    return w
end

function num_nodes(s::ComplexityTrackerState)
    return length(s.nodes)
end

function num_edges(s::ComplexityTrackerState)
    return length(s.edges)
end

function hw_composite(s::ComplexityTrackerState)
    time_score = min(1.0, 100.0 / max(1.0, s.fmax_mhz)) if s.fmax_mhz > 0 else 0.0
    return (
        0.5 * s.fpga_accuracy
        + 0.3 * (1.0 - time_score)
        + 0.2 * (1.0 if s.timing_met else 0.0)
    )
end

function submit(s::ComplexityTrackerState, report)
    s.reports[report.genome_id] = report
end

function get(s::ComplexityTrackerState, genome_id)
    return s.reports.get(genome_id)
end

function total_reports(s::ComplexityTrackerState)
    return length(s.reports)
end

function record(s::ComplexityTrackerState, stats)
    s.history = push!(, stats)
end

function generations_tracked(s::ComplexityTrackerState)
    return length(s.history)
end

function fitness_trajectory(s::ComplexityTrackerState)
    return [s.best_fitness for s in s.history]
end

function diversity_trajectory(s::ComplexityTrackerState)
    return [s.diversity for s in s.history]
end

function improvement_rate(s::ComplexityTrackerState)
    if length(s.history) < 2
        return 0.0
    return s.history[-1].best_fitness - s.history[0].best_fitness
end

function is_identical(s::ComplexityTrackerState)
    return s.total_param_changes == 0
end

function genome_diff(a, b)
    va, vb = a.to_vector(), b.to_vector()
    changes = int(sum(abs(va - vb) > 1e-8))
    return GenomeDiff(
        neuron_delta=b.topology.num_neurons - a.topology.num_neurons,
        layer_delta=b.topology.num_layers - a.topology.num_layers,
        connectivity_delta=b.topology.connectivity - a.topology.connectivity,
        tau_fast_delta=b.neuron.tau_fast - a.neuron.tau_fast,
        tau_deep_delta=b.neuron.tau_deep - a.neuron.tau_deep,
        total_param_changes=changes,
    )
end

function genome_complexity(genome)
    v = genome.to_vector()
    v_norm = v / (abs(v).max() + 1e-10)
    v_pos = abs(v_norm) + 1e-10
    v_pos = v_pos / v_pos.sum()
    entropy = -float(sum(v_pos * np.log2(v_pos)))
    topology_complexity = (
        genome.topology.num_neurons * genome.topology.num_layers * genome.topology.connectivity
    )
    return entropy + np.log2(1 + topology_complexity)
end

function record(s::ComplexityTrackerState, generation, population)
    if ! population
        return
    complexities = [genome_complexity(o.genome) for o in population]
    s.history = push!(, 
        (
            generation,
            float(mean(complexities)),
            float(np.max(complexities)),
        )
    )
end

function mean_trajectory(s::ComplexityTrackerState)
    return [h[1] for h in s.history]
end

function is_complexifying(s::ComplexityTrackerState)
    if length(s.history) < 3
        return false
    return s.history[-1][1] > s.history[0][1]
end

end # module EvoSubstrateAccel
