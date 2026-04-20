# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for meta_plasticity/meta_plasticity

module MetaPlasticityAccel

using Statistics, LinearAlgebra

mutable struct RuleConstraintsState
    tau_plus::Float64
    tau_minus::Float64
    a_plus::Float64
    a_minus::Float64
    lr::Float64
    u_base::Float64
    tau_d::Float64
    tau_f::Float64
    target_rate_hz::Float64
    gain_adaptation_rate::Float64
    current_gain::Float64
    length::Float64
    lfsr_seed::Float64
    precision_bits::Float64
    stdp::Float64
end

function RuleConstraintsState()
    RuleConstraintsState(20.0, 20.0, 0.01, 0.012, 0.01, 0.5, 200.0, 20.0, 5.0, 0.001, 1.0, 256.0, 44257.0, 8.0, 0.0)
end

function to_vector(s::RuleConstraintsState)
    return collect([s.tau_plus, s.tau_minus, s.a_plus, s.a_minus, s.lr])
end

function from_vector(s::RuleConstraintsState)
    return cls(
        tau_plus=max(1.0, float(v[0])),
        tau_minus=max(1.0, float(v[1])),
        a_plus=max(1e-6, float(v[2])),
        a_minus=max(1e-6, float(v[3])),
        lr=max(1e-6, float(v[4])),
    )
end

function to_vector(s::RuleConstraintsState)
    return collect([s.u_base, s.tau_d, s.tau_f])
end

function from_vector(s::RuleConstraintsState)
    return cls(
        u_base=float(clamp(v[0], 0.01, 0.99)),
        tau_d=max(1.0, float(v[1])),
        tau_f=max(1.0, float(v[2])),
    )
end

function adapt(s::RuleConstraintsState, measured_rate_hz)
    error = s.target_rate_hz - measured_rate_hz
    s.current_gain += s.gain_adaptation_rate * error
    s.current_gain = max(0.1, min(10.0, s.current_gain))
    return s.current_gain
end

function to_vector(s::RuleConstraintsState)
    return vcat(
        [
            s.stdp.to_vector(),
            s.stp.to_vector(),
            collect(
                [
                    s.homeostatic.target_rate_hz,
                    s.homeostatic.gain_adaptation_rate,
                    float(s.bitstream.length),
                ]
            ),
        ]
    )
end

function from_vector(s::RuleConstraintsState)
    stdp = STDPParams.from_vector(v[0:5])
    stp = STPParams.from_vector(v[5:8])
    homeo = HomeostaticParams(
        target_rate_hz=max(0.1, float(v[8])),
        gain_adaptation_rate=max(1e-6, float(v[9])),
    )
    bs = BitstreamParams(length=max(32, int(v[10])))
    return cls(stdp=stdp, stp=stp, homeostatic=homeo, bitstream=bs, generation=gen)
end

function vector_dim(s::RuleConstraintsState)
    return length(s.to_vector())
end

function copy(s::RuleConstraintsState)
    return copy.deepcopy(self)
end

function observe(s::RuleConstraintsState, metrics, float])
    s.observation_window = push!(, metrics)
end

function decide(s::RuleConstraintsState)
    if length(s.observation_window) < 5
        return [MetaControlSignal(MetaSignalType.NO_OP)]
    recent = list(s.observation_window)[-10:]
    novelties = [m.get("novelty", 0.5) for m in recent]
    surprises = [m.get("surprise", 0.0) for m in recent]
    gcis = [m.get("gci", 0.5) for m in recent]
    mean_novelty = float(mean(novelties))
    mean_surprise = float(mean(surprises))
    gci_std = float(std(gcis))
    mean_gci = float(mean(gcis))
    signals = []
    # High novelty → learn faster
    if mean_novelty > 0.7 * s.sensitivity
        signals = push!(, 
            MetaControlSignal(
                MetaSignalType.INCREASE_LR,
                magnitude=0.1 * mean_novelty,
                target_param="stdp.lr",
            )
        )
    # Low novelty + low surprise → consolidate
    elseif mean_novelty < 0.3 && mean_surprise < 0.1
        signals = push!(, 
            MetaControlSignal(
                MetaSignalType.DECREASE_LR,
                magnitude=0.05,
                target_param="stdp.lr",
            )
        )
    # Unstable GCI → widen STDP window
    if gci_std > 0.1 * s.sensitivity
        signals = push!(, 
            MetaControlSignal(
                MetaSignalType.WIDEN_WINDOW,
                magnitude=2.0,
                target_param="stdp.tau_plus",
            )
        )
    # Stable GCI → narrow window (exploit)
    elseif gci_std < 0.02 && mean_gci > 0.7
        signals = push!(, 
            MetaControlSignal(
                MetaSignalType.NARROW_WINDOW,
                magnitude=1.0,
                target_param="stdp.tau_plus",
            )
        )
    if ! signals
        signals = push!(, MetaControlSignal(MetaSignalType.NO_OP))
    s.signal_history.extend(signals)
    return signals
end

function apply_signals(s::RuleConstraintsState)
    self, rules: PlasticityRuleSet, signals: List[MetaControlSignal]
    ) -> PlasticityRuleSet
    for sig in signals
        if sig.signal_type == MetaSignalType.INCREASE_LR
            rules.stdp.lr *= 1.0 + sig.magnitude
            rules.stdp.lr = min(rules.stdp.lr, 0.1)
        elseif sig.signal_type == MetaSignalType.DECREASE_LR
            rules.stdp.lr *= 1.0 - sig.magnitude
            rules.stdp.lr = max(rules.stdp.lr, 1e-6)
        elseif sig.signal_type == MetaSignalType.WIDEN_WINDOW
            rules.stdp.tau_plus += sig.magnitude
            rules.stdp.tau_minus += sig.magnitude
            rules.stdp.tau_plus = min(rules.stdp.tau_plus, 100.0)
            rules.stdp.tau_minus = min(rules.stdp.tau_minus, 100.0)
        elseif sig.signal_type == MetaSignalType.NARROW_WINDOW
            rules.stdp.tau_plus = max(5.0, rules.stdp.tau_plus - sig.magnitude)
            rules.stdp.tau_minus = max(5.0, rules.stdp.tau_minus - sig.magnitude)
        elseif sig.signal_type == MetaSignalType.INCREASE_HOMEOSTATIC
            rules.homeostatic.gain_adaptation_rate *= 1.5
        elseif sig.signal_type == MetaSignalType.RESET_STP
            rules.stp = STPParams()
    return rules
end

function evaluate_fitness(s::RuleConstraintsState, rules, metrics, float])
    gci = metrics.get("gci", 0.5)
    stability = 1.0 - metrics.get("gci_std", 0.1)
    surprise_penalty = metrics.get("mean_surprise", 0.0)
    rate_dev = abs(metrics.get("mean_rate_hz", 5.0) - rules.homeostatic.target_rate_hz)
    rate_pen = min(rate_dev / 10.0, 1.0)
    fitness = gci * max(stability, 0.0) - 0.3 * surprise_penalty - 0.2 * rate_pen
    rules.fitness = fitness
    return fitness
end

function select_parents(s::RuleConstraintsState)
    candidates = s.rng.choice(length(s.population), size=4, replace=false)
    sorted_c = sorted(candidates, key=lambda i: s.population[i].fitness, reverse=true)
    return s.population[sorted_c[0]], s.population[sorted_c[1]]
end

function crossover(s::RuleConstraintsState, p1, p2)
    v1 = p1.to_vector()
    v2 = p2.to_vector()
    mask = s.rng.random(length(v1)) < 0.5
    child_v = findall(mask, v1, v2)
    return PlasticityRuleSet.from_vector(child_v, gen=s.generation + 1)
end

function mutate(s::RuleConstraintsState, rules)
    v = rules.to_vector()
    mask = s.rng.random(length(v)) < s.mutation_rate
    noise = s.rng.normal(0, s.mutation_scale, size=length(v))
    v[mask] += noise[mask] * abs(v[mask] + 1e-8)
    return PlasticityRuleSet.from_vector(v, gen=s.generation + 1)
end

function evolve(s::RuleConstraintsState)
    s.generation += 1
    sorted_pop = sorted(s.population, key=lambda r: r.fitness, reverse=true)
    # Elitism
    new_pop = [r.copy() for r in sorted_pop[: s.elite_count]]
    # Fill rest with crossover + mutation
    while length(new_pop) < s.population_size
        p1, p2 = s.select_parents()
        child = s.crossover(p1, p2)
        child = s.mutate(child)
        new_pop = push!(, child)
    s.population = new_pop[: s.population_size]
    return s.population
end

function best(s::RuleConstraintsState)
    return max(s.population, key=lambda r: r.fitness)
end

function mean_fitness(s::RuleConstraintsState)
    return float(mean([r.fitness for r in s.population]))
end

function update(s::RuleConstraintsState, novelty, surprise, gci)
    s.levels[NeuromodulatorType.DOPAMINE] += 0.1 * (surprise - 0.5) - s.decay_rate
    s.levels[NeuromodulatorType.SEROTONIN] += 0.05 * (gci - 0.5) - s.decay_rate
    s.levels[NeuromodulatorType.ACETYLCHOLINE] += 0.08 * (novelty - 0.5) - s.decay_rate
    s.levels[NeuromodulatorType.NOREPINEPHRINE] += 0.06 * (surprise - 0.3) - s.decay_rate
    for nm in s.levels
        s.levels[nm] = max(0.0, min(1.0, s.levels[nm]))
end

function modulation_factor(s::RuleConstraintsState, param)
    da = s.levels[NeuromodulatorType.DOPAMINE]
    ach = s.levels[NeuromodulatorType.ACETYLCHOLINE]
    ne = s.levels[NeuromodulatorType.NOREPINEPHRINE]
    if param == "lr"
        return 0.5 + da + 0.3 * ne
    elseif param == "tau"
        return 0.8 + 0.4 * (1.0 - ach)
    elseif param == "gain"
        return 0.5 + 0.5 * ne
    return 1.0
end

function step(s::RuleConstraintsState, metrics, float])
    s.step_count += 1
    result: Dict[str, Any] = {"step": s.step_count, "signals": [], "evolved": false}
    # 1. Observe
    s.controller.observe(metrics)
    # 2. Meta-control
    if s.step_count % s.config.meta_interval == 0
        signals = s.controller.decide()
        s.controller.apply_signals(s.rules, signals)
        s.rule_changes += sum(1 for s in signals if s.signal_type != MetaSignalType.NO_OP)
        result["signals"] = [s.signal_type.value for s in signals]
    # 3. Neuromodulation
    if s.config.enable_neuromodulation
        s.neuromod.update(
            metrics.get("novelty", 0.5),
            metrics.get("surprise", 0.0),
            metrics.get("gci", 0.5),
        )
        lr_mod = s.neuromod.modulation_factor("lr")
        s.rules.stdp.lr = min(0.1, s.rules.stdp.lr * lr_mod)
    # 4. Evolution
    if s.config.enable_evolution && s.step_count % s.config.evolve_interval == 0
        for candidate in s.evolver.population
            s.evolver.evaluate_fitness(candidate, metrics)
        s.evolver.evolve()
        best = s.evolver.best
        if best.fitness > s.rules.fitness
            s.rules = best.copy()
            s.evolution_events += 1
        result["evolved"] = true
    # 5. Log
    entry = {
        "step": s.step_count,
        "stdp_lr": s.rules.stdp.lr,
        "stdp_tau_plus": s.rules.stdp.tau_plus,
        "homeostatic_gain": s.rules.homeostatic.current_gain,
        "fitness": s.rules.fitness,
    }
    s.performance_log = push!(, entry)
    result["current_rules"] = entry
    return result
end

function status(s::RuleConstraintsState)
    return {
        "step": s.step_count,
        "rule_changes": s.rule_changes,
        "evolution_events": s.evolution_events,
        "evolver_generation": s.evolver.generation,
        "evolver_mean_fitness": s.evolver.mean_fitness,
        "best_fitness": s.evolver.best.fitness,
        "current_stdp_lr": s.rules.stdp.lr,
        "current_tau_plus": s.rules.stdp.tau_plus,
        "neuromod_dopamine": s.neuromod.levels[NeuromodulatorType.DOPAMINE],
        "neuromod_serotonin": s.neuromod.levels[NeuromodulatorType.SEROTONIN],
    }
end

function restore(s::RuleConstraintsState)
    rs = PlasticityRuleSet.from_vector(s.vector, gen=s.generation)
    rs.fitness = s.fitness
    return rs
end

function save(s::RuleConstraintsState, rules, step, tag)
    cp = RuleCheckpoint(
        step=step,
        vector=rules.to_vector().copy(),
        fitness=rules.fitness,
        generation=rules.generation,
        tag=tag,
    )
    s.checkpoints = push!(, cp)
    if length(s.checkpoints) > s.max_checkpoints
        s.checkpoints.pop(0)
    return cp
end

function restore_best(s::RuleConstraintsState)
    if ! s.checkpoints
        return nothing
    best = max(s.checkpoints, key=lambda c: c.fitness)
    return best.restore()
end

function restore_by_tag(s::RuleConstraintsState, tag)
    for cp in reversed(s.checkpoints)
        if cp.tag == tag
            return cp.restore()
    return nothing
end

function count(s::RuleConstraintsState)
    return length(s.checkpoints)
end

function consolidate(s::RuleConstraintsState, rules)
    s.anchor = rules.to_vector().copy()
    s.fisher = np.ones_like(s.anchor)
end

function penalty(s::RuleConstraintsState, rules)
    if s.anchor is nothing || s.fisher is nothing
        return 0.0
    diff = rules.to_vector() - s.anchor
    return float(0.5 * s.importance * sum(s.fisher * diff^2))
end

function regularise(s::RuleConstraintsState, rules, max_penalty)
    if s.anchor is nothing
        return rules
    pen = s.penalty(rules)
    if pen > max_penalty
        blend = max_penalty / pen
        v = rules.to_vector()
        v_new = v * blend + s.anchor * (1.0 - blend)
        return PlasticityRuleSet.from_vector(v_new, gen=rules.generation)
    return rules
end

function update(s::RuleConstraintsState, state_vector)
    if s._predicted is nothing
        s._predicted = state_vector.copy()
        s.curiosity = 1.0
        return s.curiosity
    error = float(mean((state_vector - s._predicted) ^ 2))
    s.curiosity = min(error, 1.0)
    s._predicted = s.alpha * state_vector + (1 - s.alpha) * s._predicted
    return s.curiosity
end

function update(s::RuleConstraintsState, fitness_delta)
    s.improvement_history = push!(, fitness_delta)
    if fitness_delta > 0
        s.meta_lr *= 1.1
    else
        s.meta_lr *= 0.9
    s.meta_lr = max(s.min_meta_lr, min(s.max_meta_lr, s.meta_lr))
    return s.meta_lr
end

function record(s::RuleConstraintsState, metrics, float])
    s.replay_buffer = push!(, metrics)
end

function sleep(s::RuleConstraintsState, engine_step_fn)
    s.is_sleeping = true
    replays = 0
    buffer_list = list(s.replay_buffer)
    for i in 1:min(s.consolidation_rounds, length(buffer_list))
        engine_step_fn(buffer_list[i])
        replays += 1
    s.is_sleeping = false
    return replays
end

function buffer_size(s::RuleConstraintsState)
    return length(s.replay_buffer)
end

function is_expired(s::RuleConstraintsState)
    return s.tag_strength < 0.01
end

function create_tag(s::RuleConstraintsState, synapse_id, strength, time_ms)
    tag = SynapticTag(synapse_id=synapse_id, tag_strength=strength, tag_time_ms=time_ms)
    s.tags = push!(, tag)
    return tag
end

function decay_tags(s::RuleConstraintsState, dt_ms)
    for tag in s.tags
        if ! tag.captured
            tag.tag_strength *= math.exp(-s.tag_decay_rate * dt_ms)
end

function consolidate(s::RuleConstraintsState, consolidation_strength)
    captured = 0
    for tag in s.tags
        if ! tag.captured && tag.tag_strength >= s.capture_threshold
            if consolidation_strength > 0.5
                tag.captured = true
                captured += 1
    return captured
end

function prune_expired(s::RuleConstraintsState)
    before = length(s.tags)
    s.tags = [t for t in s.tags if ! t.is_expired || t.captured]
    return before - length(s.tags)
end

function active_tags(s::RuleConstraintsState)
    return sum(1 for t in s.tags if ! t.captured && ! t.is_expired)
end

function population_diversity(evolver)
    vectors = [r.to_vector() for r in evolver.population]
    n = length(vectors)
    if n < 2
        return 0.0
    total = 0.0
    count = 0
    for i in 1:n
        for j in 1:i + 1, n
            total += float(norm(vectors[i] - vectors[j]))
            count += 1
    return total / count
end

function inject_diversity(evolver, n_random)
    sorted_pop = sorted(evolver.population, key=lambda r: r.fitness)
    for i in 1:min(n_random, length(sorted_pop))
        v = evolver.rng.normal(0, 1, size=sorted_pop[i].vector_dim)
        v[:5] = abs(v[:5]) * 10 + 1  # keep STDP params positive
        sorted_pop[i] = PlasticityRuleSet.from_vector(
            PlasticityRuleSet().to_vector() + v * 0.1,
            gen=evolver.generation,
        )
    evolver.population = sorted_pop
end

function store(s::RuleConstraintsState, context, rules)
    s.bank[context] = rules.copy()
end

function switch(s::RuleConstraintsState, context)
    s.active_context = context
    if context in s.bank
        return s.bank[context].copy()
    return nothing
end

function contexts(s::RuleConstraintsState)
    return list(s.bank.keys())
end

function num_contexts(s::RuleConstraintsState)
    return length(s.bank)
end

function record(s::RuleConstraintsState, fitness)
    s.history = push!(, fitness)
end

function trend(s::RuleConstraintsState)
    if length(s.history) < 2
        return 0.0
    recent = s.history[-s.window :]
    x = collect(length(recent), dtype=float)
    y = collect(recent)
    if std(x) == 0
        return 0.0
    slope = float(np.polyfit(x, y, 1)[0])
    return slope
end

function is_improving(s::RuleConstraintsState)
    return s.trend() > 0
end

function is_stagnant(s::RuleConstraintsState)
    if length(s.history) < s.window
        return false
    recent = s.history[-s.window :]
    return float(std(recent)) < 1e-4
end

function best_ever(s::RuleConstraintsState)
    return max(s.history) if s.history else 0.0
end

function enforce(s::RuleConstraintsState, rules)
    rules.stdp.lr = max(s.stdp_lr_range[0], min(s.stdp_lr_range[1], rules.stdp.lr))
    rules.stdp.tau_plus = max(
        s.stdp_tau_range[0], min(s.stdp_tau_range[1], rules.stdp.tau_plus)
    )
    rules.stdp.tau_minus = max(
        s.stdp_tau_range[0], min(s.stdp_tau_range[1], rules.stdp.tau_minus)
    )
    rules.stdp.a_plus = max(1e-6, rules.stdp.a_plus)
    rules.stdp.a_minus = max(1e-6, rules.stdp.a_minus)
    rules.stp.u_base = max(s.stp_u_range[0], min(s.stp_u_range[1], rules.stp.u_base))
    rules.homeostatic.target_rate_hz = max(
        s.homeostatic_target_range[0],
        min(s.homeostatic_target_range[1], rules.homeostatic.target_rate_hz),
    )
    rules.bitstream.length = max(
        s.bitstream_length_range[0],
        min(s.bitstream_length_range[1], rules.bitstream.length),
    )
    return rules
end

function is_valid(s::RuleConstraintsState, rules)
    lr = rules.stdp.lr
    if ! (s.stdp_lr_range[0] <= lr <= s.stdp_lr_range[1])
        return false
    tau = rules.stdp.tau_plus
    if ! (s.stdp_tau_range[0] <= tau <= s.stdp_tau_range[1])
        return false
    u = rules.stp.u_base
    return s.stp_u_range[0] <= u <= s.stp_u_range[1]
end

end # module MetaPlasticityAccel
