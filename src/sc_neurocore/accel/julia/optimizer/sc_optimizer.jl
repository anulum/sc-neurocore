# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for optimizer/sc_optimizer

module ScOptimizerAccel

using Statistics, LinearAlgebra

mutable struct SCOptimizerState
    max_luts::Float64
    max_power_mw::Float64
    max_latency_cycles::Float64
    id::Float64
    mac_count::Float64
    is_critical_path::Float64
    bitstream_length::Float64
    decorrelator::Float64
    mode::Float64
    luts_used::Float64
    power_used::Float64
    accuracy_score::Float64
    latency_cycles::Float64
    config::Float64
    total_luts::Float64
end

function SCOptimizerState()
    SCOptimizerState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function summary(s::SCOptimizerState)
    lines = [
        f"LUTs: {s.total_luts}, Power: {s.total_power_mw:.2f} mW, "
        f"Latency: {s.total_latency_cycles} cycles, "
        f"Accuracy: {s.mean_accuracy:.4f}",
    ]
    for lid, cfg in s.config.items()
        lines = push!(,
            f"  {lid}: N={cfg.bitstream_length}, "
            f"decorr={cfg.decorrelator}, mode={cfg.mode}, "
            f"acc={cfg.accuracy_score:.4f}"
        )
    return "\n".join(lines)
end

function _estimate_resources(s::SCOptimizerState)
    self,
    mac_count: int,
    length: int,
    decorr: str,
    mode: str,
    ) -> Tuple[int, float, float, int]
    if mode == "Deterministic"
        luts = mac_count * 120
        power = mac_count * 0.5
        return luts, power, 1.0, 1
    if mode == "Hybrid"
        sc_frac = 0.7
        det_frac = 0.3
        sc_luts = int(mac_count * sc_frac) * 2 + int(math.log2(length)) * 5
        det_luts = int(mac_count * det_frac) * 120
        luts = sc_luts + det_luts
        power = (mac_count * sc_frac * 0.01 * (length / 256)
                 + mac_count * det_frac * 0.5)
        accuracy = 0.95  # hybrid baseline
        latency = length
        if decorr == "Sobol"
            luts += int(mac_count * sc_frac * 15)
            accuracy = 0.97
        elseif decorr == "LFSR"
            luts += 16
            accuracy = 0.96
        return luts, power, min(1.0, accuracy), latency
    # SC mode
    luts = mac_count * 2 + int(math.log2(length)) * 5
    power = mac_count * 0.01 * (length / 256)
    latency = length
    if decorr == "Sobol"
        luts += mac_count * 15
        accuracy = 1.0 - (1.0 / length)
    elseif decorr == "Halton"
        luts += mac_count * 12
        accuracy = 1.0 - (1.2 / length)
    elseif decorr == "SCC_Decorrelator"
        luts += mac_count * 8
        accuracy = 1.0 - (1.5 / length)
    elseif decorr == "LFSR"
        luts += 16
        accuracy = 1.0 - (1.0 / math.sqrt(length))
    else
        accuracy = 1.0 - (2.0 / math.sqrt(length))
    accuracy = max(0.1, min(1.0, accuracy))
    return luts, power, accuracy, latency
end

function _generate_candidates(s::SCOptimizerState, layer)
    candidates = []
    for mode in s.modes
        if mode == "Deterministic"
            l, p, a, lat = s._estimate_resources(layer.mac_count, 1, "nothing", mode)
            candidates = push!(, LayerConfig(1, "nothing", mode, l, p, a, lat))
            continue
        for length in s.bitstream_options
            for decorr in s.decorrelators
                l, p, a, lat = s._estimate_resources(
                    layer.mac_count, length, decorr, mode
                )
                candidates = push!(, LayerConfig(length, decorr, mode, l, p, a, lat))
    return candidates
end

function _is_feasible(s::SCOptimizerState)
    self, config: Dict[str, LayerConfig]
    ) -> bool
    total_luts = sum(c.luts_used for c in config.values())
    total_power = sum(c.power_used for c in config.values())
    total_latency = max((c.latency_cycles for c in config.values()), default=0)
    if total_luts > s.budget.max_luts
        return false
    if total_power > s.budget.max_power_mw
        return false
    if s.budget.max_latency_cycles > 0 && total_latency > s.budget.max_latency_cycles
        return false
    return true
end

function _score(s::SCOptimizerState)
    self, config: Dict[str, LayerConfig], network: List[LayerProfile]
    ) -> float
    total = 0.0
    weight_sum = 0.0
    for layer in network
        w = 2.0 if layer.is_critical_path else 1.0
        total += config[layer.id].accuracy_score * w
        weight_sum += w
    return total / weight_sum if weight_sum > 0 else 0.0
end

function _build_report(s::SCOptimizerState)
    self,
    config: Dict[str, LayerConfig],
    network: List[LayerProfile],
    pareto: List[Tuple[int, float, float]] | nothing = nothing,
    ) -> OptimizerReport
    total_luts = sum(c.luts_used for c in config.values())
    total_power = sum(c.power_used for c in config.values())
    total_latency = max((c.latency_cycles for c in config.values()), default=0)
    mean_acc = s._score(config, network)
    return OptimizerReport(
        config=config,
        total_luts=total_luts,
        total_power_mw=total_power,
        total_latency_cycles=total_latency,
        mean_accuracy=mean_acc,
        pareto_frontier=pareto || [],
    )
end

function optimize(s::SCOptimizerState, network)
    current_config: Dict[str, LayerConfig] = {}
    candidates_per_layer = {
        layer.id: s._generate_candidates(layer) for layer in network
    }
    for layer in network
        cheapest = min(candidates_per_layer[layer.id], key=lambda c: c.luts_used)
        current_config[layer.id] = cheapest
    if ! s._is_feasible(current_config)
        return nothing
    upgraded = true
    while upgraded
        upgraded = false
        best_upgrade = nothing
        best_layer_id = nothing
        max_efficiency = 0.0
        for layer in network
            curr = current_config[layer.id]
            for cand in candidates_per_layer[layer.id]
                if cand.accuracy_score <= curr.accuracy_score
                    continue
                trial = dict(current_config)
                trial[layer.id] = cand
                if ! s._is_feasible(trial)
                    continue
                lut_diff = cand.luts_used - curr.luts_used
                score_gain = cand.accuracy_score - curr.accuracy_score
                if layer.is_critical_path
                    score_gain *= 2.0
                eff = score_gain / lut_diff if lut_diff > 0 else float("inf")
                if eff > max_efficiency
                    max_efficiency = eff
                    best_upgrade = cand
                    best_layer_id = layer.id
        if best_upgrade
            current_config[best_layer_id] = best_upgrade
            upgraded = true
    return s._build_report(current_config, network)
end

function optimize_annealing(s::SCOptimizerState)
    self,
    network: List[LayerProfile],
    *,
    t_init: float = 1.0,
    t_min: float = 0.001,
    alpha: float = 0.95,
    max_iter: int = 2000,
    seed: int = 42,
    ) -> Optional[OptimizerReport]
    if _HAS_RUST
        return s._optimize_annealing_rust(
            network, t_init=t_init, t_min=t_min,
            alpha=alpha, max_iter=max_iter, seed=seed,
        )
    return s._optimize_annealing_python(
        network, t_init=t_init, t_min=t_min,
        alpha=alpha, max_iter=max_iter, seed=seed,
    )
end

function _optimize_annealing_rust(s::SCOptimizerState)
    self,
    network: List[LayerProfile],
    *,
    t_init: float = 1.0,
    t_min: float = 0.001,
    alpha: float = 0.95,
    max_iter: int = 2000,
    seed: int = 42,
    ) -> Optional[OptimizerReport]
    mac_counts = [layer.mac_count for layer in network]
    weights = [2.0 if layer.is_critical_path else 1.0 for layer in network]
    result = py_opt_sa_search(
        mac_counts, weights,
        s.budget.max_luts, s.budget.max_power_mw,
        s.budget.max_latency_cycles,
        t_init, t_min, alpha, max_iter, seed,
    )
    if ! result.get("feasible", false)
        return nothing
    layer_luts = result["layer_luts"]
    layer_power = result["layer_power"]
    layer_accuracy = result["layer_accuracy"]
    config: Dict[str, LayerConfig] = {}
    for i, layer in enumerate(network)
        config[layer.id] = LayerConfig(
            bitstream_length=0,
            decorrelator="auto",
            mode="auto",
            luts_used=layer_luts[i],
            power_used=layer_power[i],
            accuracy_score=layer_accuracy[i],
        )
    pareto_luts = result.get("pareto_luts", [])
    pareto_power = result.get("pareto_power", [])
    pareto_score = result.get("pareto_score", [])
    if pareto_luts
        pareto_result = py_opt_extract_pareto(
            pareto_luts, pareto_power, pareto_score,
        )
        frontier = list(zip(
            pareto_result["luts"],
            pareto_result["power"],
            pareto_result["score"],
        ))
    else
        frontier = []
    return s._build_report(config, network, frontier)
end

function _optimize_annealing_python(s::SCOptimizerState)
    self,
    network: List[LayerProfile],
    *,
    t_init: float = 1.0,
    t_min: float = 0.001,
    alpha: float = 0.95,
    max_iter: int = 2000,
    seed: int = 42,
    ) -> Optional[OptimizerReport]
    rng = random.Random(seed)
    candidates_per_layer = {
        layer.id: s._generate_candidates(layer) for layer in network
    }
    current: Dict[str, LayerConfig] = {}
    for layer in network
        cheapest = min(candidates_per_layer[layer.id], key=lambda c: c.luts_used)
        current[layer.id] = cheapest
    if ! s._is_feasible(current)
        return nothing
    best = dict(current)
    best_score = s._score(best, network)
    current_score = best_score
    t = t_init
    pareto_points: List[Tuple[int, float, float]] = []
    while t > t_min && max_iter > 0
        max_iter -= 1
        layer = rng.choice(network)
        cand = rng.choice(candidates_per_layer[layer.id])
        trial = dict(current)
        trial[layer.id] = cand
        if ! s._is_feasible(trial)
            t *= alpha
            continue
        trial_score = s._score(trial, network)
        delta = trial_score - current_score
        if delta > 0 || rng.random() < math.exp(delta / t)
            current = trial
            current_score = trial_score
            if current_score > best_score
                best = dict(current)
                best_score = current_score
            luts = sum(c.luts_used for c in current.values())
            power = sum(c.power_used for c in current.values())
            pareto_points = push!(, (luts, power, current_score))
        t *= alpha
    frontier = s._extract_pareto(pareto_points)
    return s._build_report(best, network, frontier)
end

function _extract_pareto(s::SCOptimizerState)
    points: List[Tuple[int, float, float]],
    ) -> List[Tuple[int, float, float]]
    if ! points
        return []
    frontier = []
    for p in points
        dominated = false
        for q in points
            if q is p
                continue
            # q dominates p if q uses ≤ resources AND has ≥ accuracy
            if q[0] <= p[0] && q[1] <= p[1] && q[2] >= p[2]
                if q[0] < p[0] || q[1] < p[1] || q[2] > p[2]
                    dominated = true
                    break
        if ! dominated
            frontier = push!(, p)
    # Sort by LUTs ascending
    frontier.sort(key=lambda x: x[0])
    # Deduplicate
    seen = set()
    deduped = []
    for pt in frontier
        key = (pt[0], round(pt[1], 4), round(pt[2], 4))
        if key ! in seen
            seen.add(key)
            deduped = push!(, pt)
    return deduped
end

end # module ScOptimizerAccel
