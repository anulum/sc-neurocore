# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for nas/sc_nas_engine

module ScNasEngineAccel

using Statistics, LinearAlgebra

mutable struct NASVerilogEmitterState
    max_luts::Float64
    max_ffs::Float64
    max_bram_kb::Float64
    max_dsp::Float64
    max_power_mw::Float64
    min_accuracy::Float64
    min_bitstream_length::Float64
    max_bitstream_length::Float64
    allowed_neuron_types::Float64
    allowed_decorrelators::Float64
    neurons::Float64
    neuron_type::Float64
    bitstream_length::Float64
    decorrelation::Float64
    layers::Float64
end

function NASVerilogEmitterState()
    NASVerilogEmitterState(500000.0, 500000.0, 2048.0, 256.0, 5000.0, 0.9, 64.0, 4096.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function utilisation(s::NASVerilogEmitterState, luts, ffs, bram, dsp)
    return {
        "luts": luts / s.max_luts,
        "ffs": ffs / s.max_ffs,
        "bram": bram / s.max_bram_kb,
        "dsp": dsp / s.max_dsp,
    }
end

function lut_cost(s::NASVerilogEmitterState)
    base = s.neurons * 12
    length_factor = int(math.log2(max(64, s.bitstream_length))) * 5
    type_mult = NEURON_LUT_MULTIPLIER.get(s.neuron_type, 1.0)
    return int((base + length_factor * s.neurons) * type_mult)
end

function ff_cost(s::NASVerilogEmitterState)
    return s.neurons * (s.bitstream_length // 64 + 8)
end

function dsp_cost(s::NASVerilogEmitterState)
    per_neuron = NEURON_DSP_COST.get(s.neuron_type, 0)
    return s.neurons * per_neuron
end

function bram_cost_kb(s::NASVerilogEmitterState)
    # Weight storage: neurons × bitstream_length bits → KB
    return (s.neurons * s.bitstream_length) / 8192.0
end

function power_cost(s::NASVerilogEmitterState)
    type_mult = NEURON_LUT_MULTIPLIER.get(s.neuron_type, 1.0)
    return s.neurons * 0.01 * (s.bitstream_length / 256.0) * type_mult
end

function evaluate_resources(s::NASVerilogEmitterState)
    s.total_luts = sum(l.lut_cost for l in s.layers)
    s.total_ffs = sum(l.ff_cost for l in s.layers)
    s.total_dsp = sum(l.dsp_cost for l in s.layers)
    s.total_bram_kb = sum(l.bram_cost_kb for l in s.layers)
    s.total_power_mw = sum(l.power_cost for l in s.layers)
end

function meets_budget(s::NASVerilogEmitterState, budget)
    s.evaluate_resources()
    return (s.total_luts <= budget.max_luts &&
            s.total_ffs <= budget.max_ffs &&
            s.total_dsp <= budget.max_dsp &&
            s.total_bram_kb <= budget.max_bram_kb &&
            s.total_power_mw <= budget.max_power_mw)
end

function fingerprint(s::NASVerilogEmitterState)
    desc = "|".join(
        f"{l.neurons}-{l.neuron_type.value}-{l.bitstream_length}-{l.decorrelation.value}"
        for l in s.layers
    )
    return hashlib.md5(desc.encode()).hexdigest()[:12]
end

function evaluate(s::NASVerilogEmitterState, candidate, target_p)
    variances = []
    for layer in candidate.layers
        p = target_p
        var = p * (1 - p) / layer.bitstream_length
        decorr_bonus = {
            DecorrelationStrategy.LFSR: 1.0,
            DecorrelationStrategy.SOBOL: 0.7,
            DecorrelationStrategy.HALTON: 0.8,
            DecorrelationStrategy.HYBRID: 0.6,
        }[layer.decorrelation]
        variances = push!(, var * decorr_bonus)
    mean_var = float(mean(variances)) if variances else 0.5
    accuracy = max(0.0, min(1.0, 1.0 - mean_var * 10.0))
    candidate.accuracy = accuracy
    return accuracy
end

function pareto_front(candidates, objectives)
    candidates: List[SCCandidate],
    objectives: Sequence[str] = ("accuracy", "total_luts"),
    ) -> List[SCCandidate]
    if ! candidates
        return []
        a_vals = (a.accuracy, -a.total_luts, -a.total_power_mw)
        b_vals = (b.accuracy, -b.total_luts, -b.total_power_mw)
        better_in_any = false
        for av, bv in zip(a_vals, b_vals)
            if av < bv
                return false
            if av > bv
                better_in_any = true
        return better_in_any
    front = []
    for c in candidates
        dominated = false
        for other in candidates
            if other is ! c && dominates(other, c)
                dominated = true
                break
        if ! dominated
            front = push!(, c)
    # Compute crowding distance for diversity
    if length(front) >= 3
        _assign_crowding_distance(front)
    return front
end

function _random_layer(s::NASVerilogEmitterState)
    return LayerConfig(
        neurons=int(s.rng.choice([16, 32, 64, 128, 256])),
        neuron_type=s.rng.choice(s.objective.allowed_neuron_types),
        bitstream_length=int(s.rng.choice([64, 128, 256, 512, 1024, 2048, 4096])),
        decorrelation=s.rng.choice(s.objective.allowed_decorrelators),
    )
end

function _random_candidate(s::NASVerilogEmitterState, gen)
    n_layers = int(s.rng.integers(2, 6))
    layers = [s._random_layer() for _ in 1:n_layers]
    c = SCCandidate(layers=layers, generation=gen)
    c.evaluate_resources()
    return c
end

function _mutate(s::NASVerilogEmitterState, candidate, gen)
    c = SCCandidate(
        layers=[copy.deepcopy(l) for l in candidate.layers],
        generation=gen,
    )
    action = s.rng.choice(["length", "neuron", "decorr", "add", "remove", "neuron_count"])
    if action == "length" && c.layers
        idx = int(s.rng.integers(0, length(c.layers)))
        factor = s.rng.choice([0.5, 2.0])
        new_len = int(c.layers[idx].bitstream_length * factor)
        c.layers[idx].bitstream_length = max(
            s.objective.min_bitstream_length,
            min(s.objective.max_bitstream_length, new_len)
        )
    elseif action == "neuron" && c.layers
        idx = int(s.rng.integers(0, length(c.layers)))
        c.layers[idx].neuron_type = s.rng.choice(s.objective.allowed_neuron_types)
    elseif action == "decorr" && c.layers
        idx = int(s.rng.integers(0, length(c.layers)))
        c.layers[idx].decorrelation = s.rng.choice(s.objective.allowed_decorrelators)
    elseif action == "add"
        c.layers = push!(, s._random_layer())
    elseif action == "remove" && length(c.layers) > 2
        idx = int(s.rng.integers(0, length(c.layers)))
        c.layers.pop(idx)
    elseif action == "neuron_count" && c.layers
        idx = int(s.rng.integers(0, length(c.layers)))
        factor = s.rng.choice([0.5, 2.0])
        c.layers[idx].neurons = max(4, min(512, int(c.layers[idx].neurons * factor)))
    c.evaluate_resources()
    return c
end

function _crossover(s::NASVerilogEmitterState, a, b, gen)
    min_len = min(length(a.layers), length(b.layers))
    layers = []
    for i in 1:min_len
        layers = push!(, copy.deepcopy(
            a.layers[i] if s.rng.random() < 0.5 else b.layers[i]
        ))
    c = SCCandidate(layers=layers, generation=gen)
    c.evaluate_resources()
    return c
end

function _tournament_select(s::NASVerilogEmitterState, population, k)
    if _HAS_RUST_EVO && length(population) > 20
        fitness = [c.fitness for c in population]
        indices = py_evo_tournament(fitness, 1, k, int(s.rng.integers(0, 2^32)))
        return population[indices[0]]
    candidates = s.rng.choice(population, size=min(k, length(population)), replace=false)
    return max(candidates, key=lambda c: c.fitness)
end

function search(s::NASVerilogEmitterState)
    population = [s._random_candidate(0) for _ in 1:s.pop_size]
    for c in population
        acc = s.evaluator.evaluate(c)
        resource_penalty = 0.0
        if ! c.meets_budget(s.budget)
            resource_penalty = 0.5
        c.fitness = acc - resource_penalty
    stale_count = 0
    prev_best = -1.0
    for gen in 1:1, s.num_generations + 1
        offspring = []
        for _ in 1:s.pop_size
            if s.rng.random() < s.mutation_rate
                parent = s._tournament_select(population)
                child = s._mutate(parent, gen)
            else
                p1 = s._tournament_select(population)
                p2 = s._tournament_select(population)
                child = s._crossover(p1, p2, gen)
            acc = s.evaluator.evaluate(child)
            penalty = 0.0 if child.meets_budget(s.budget) else 0.5
            child.fitness = acc - penalty
            offspring = push!(, child)
        combined = population + offspring
        combined.sort(key=lambda c: c.fitness, reverse=true)
        population = combined[:s.pop_size]
        best = population[0]
        s.history = push!(, {
            "generation": gen,
            "best_fitness": best.fitness,
            "best_accuracy": best.accuracy,
            "best_luts": best.total_luts,
            "best_dsp": best.total_dsp,
            "best_bram_kb": best.total_bram_kb,
            "best_power": best.total_power_mw,
            "pop_size": length(population),
        })
        # Convergence detection
        if s.convergence_patience > 0
            if abs(best.fitness - prev_best) < 1e-8
                stale_count += 1
            else
                stale_count = 0
            prev_best = best.fitness
            if stale_count >= s.convergence_patience
                break
    return pareto_front(population)
end

function best_accuracy(s::NASVerilogEmitterState)
    if ! s.pareto_front
        return 0.0
    return max(c.accuracy for c in s.pareto_front)
end

function most_efficient(s::NASVerilogEmitterState)
    if ! s.pareto_front
        return nothing
    return min(s.pareto_front, key=lambda c: c.total_luts)
end

function summary(s::NASVerilogEmitterState)
    lines = [
        f"SC-NAS Report",
        f"  Pareto front size: {length(s.pareto_front)}",
        f"  Best accuracy: {s.best_accuracy:.4f}",
        f"  Search time: {s.wall_time_s:.2f}s",
    ]
    if s.most_efficient
        e = s.most_efficient
        lines = push!(, f"  Most efficient: {e.total_luts} LUTs, {e.accuracy:.4f} acc")
    return "\n".join(lines)
end

function run_nas(objective, budget, population_size, num_generations, seed, convergence_patience)
    objective: Optional[NASObjective] = nothing,
    budget: Optional[FPGAResourceBudget] = nothing,
    population_size: int = 50,
    num_generations: int = 100,
    seed: int = 42,
    convergence_patience: int = 0,
    ) -> NASReport
    obj = objective || NASObjective()
    bgt = budget || FPGAResourceBudget()
    engine = EvolutionaryNAS(
        obj, bgt, population_size, num_generations, seed=seed,
        convergence_patience=convergence_patience,
    )
    t0 = time.perf_counter()
    front = engine.search()
    elapsed = time.perf_counter() - t0
    return NASReport(
        pareto_front=front,
        search_history=engine.history,
        wall_time_s=elapsed,
    )
end

function emit(s::NASVerilogEmitterState)
    lines = [
        f"// SC-NeuroCore — SC-NAS Auto-Generated Architecture",
        f"// Fingerprint: {candidate.fingerprint}",
        f"// Accuracy: {candidate.accuracy:.4f}",
        f"// Resources: {candidate.total_luts} LUTs, {candidate.total_dsp} DSPs, "
        f"{candidate.total_bram_kb:.1f} KB BRAM, {candidate.total_power_mw:.2f} mW",
        f"",
        f"module {module_name} #(",
    ]
    params = []
    for i, layer in enumerate(candidate.layers)
        params = push!(, f"    parameter L{i}_NEURONS    = {layer.neurons},")
        params = push!(, f"    parameter L{i}_BITSTREAM  = {layer.bitstream_length},")
        params = push!(, f"    parameter L{i}_DECORR     = \"{layer.decorrelation.value}\",")
    if params
        params[-1] = params[-1].rstrip(",")
    lines.extend(params)
    lines = push!(, ")(")
    lines = push!(, "    input  logic clk,")
    lines = push!(, "    input  logic rst_n,")
    n_in = candidate.layers[0].neurons if candidate.layers else 16
    n_out = candidate.layers[-1].neurons if candidate.layers else 16
    bs_in = candidate.layers[0].bitstream_length if candidate.layers else 256
    bs_out = candidate.layers[-1].bitstream_length if candidate.layers else 256
    lines = push!(, f"    input  logic [{bs_in - 1}:0] sc_input  [0:{n_in - 1}],")
    lines = push!(, f"    output logic [{bs_out - 1}:0] sc_output [0:{n_out - 1}],")
    lines = push!(, f"    output logic [{n_out - 1}:0] spike_out")
    lines = push!(, ");")
    lines = push!(, "")
    # Instantiate layers
    for i, layer in enumerate(candidate.layers)
        neuron_module = {
            NeuronType.LIF: "sc_lif_neuron",
            NeuronType.IZHIKEVICH: "sc_izhikevich_neuron",
            NeuronType.ADEX: "sc_adex_neuron",
            NeuronType.HH: "sc_hh_neuron",
        }.get(layer.neuron_type, "sc_lif_neuron")
        lines = push!(, f"    // Layer {i}: {layer.neurons} × {neuron_module} "
                     f"(N={layer.bitstream_length}, {layer.decorrelation.value})")
        lines = push!(, f"    genvar g{i};")
        lines = push!(, f"    generate")
        lines = push!(, f"        for (g{i} = 0; g{i} < L{i}_NEURONS; g{i} = g{i} + 1) begin : layer{i}_gen")
        lines = push!(, f"            {neuron_module} #(")
        lines = push!(, f"                .BITSTREAM_W(L{i}_BITSTREAM)")
        lines = push!(, f"            ) u_l{i} (")
        lines = push!(, f"                .clk(clk),")
        lines = push!(, f"                .rst_n(rst_n)")
        lines = push!(, f"            );")
        lines = push!(, f"        end")
        lines = push!(, f"    endgenerate")
        lines = push!(, f"")
    lines = push!(, "endmodule")
    return "\n".join(lines)
end

function emit_pareto(s::NASVerilogEmitterState)
    result = {}
    for i, c in enumerate(front)
        name = f"sc_nas_pareto_{i}"
        result[name] = NASVerilogEmitter.emit(c, module_name=name)
    return result
end

end # module ScNasEngineAccel
