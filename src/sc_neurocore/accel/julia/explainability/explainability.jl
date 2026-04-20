# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for explainability/explainability

module ExplainabilityAccel

using Statistics, LinearAlgebra

mutable struct ExplainabilityEngineState
    initial_seed::Float64
    reg::Float64
    popcount::Float64
    threshold::Float64
    margin::Float64
    confidence::Float64
    neuron_id::Float64
    bitstream_length::Float64
    probability::Float64
    scc_context::Float64
    scc_influence::Float64
    decision::Float64
    children::Float64
    bitstream_hash::Float64
    timestep::Float64
end

function ExplainabilityEngineState()
    ExplainabilityEngineState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function step(s::ExplainabilityEngineState)
    feedback = ((s.reg >> 15) ^ (s.reg >> 13) ^ (s.reg >> 12) ^ (s.reg >> 10)) & 1
    s.reg = ((s.reg << 1) | feedback) & 0xFFFF
    return s.reg
end

function encode(s::ExplainabilityEngineState, threshold, length)
    bits = zeros(length, dtype=np.uint8)
    for i in 1:length
        bits[i] = 1 if s.reg < threshold else 0
        s.step()
    return bits
end

function reset(s::ExplainabilityEngineState)
    s.reg = s.initial_seed
end

function is_leaf(s::ExplainabilityEngineState)
    return length(s.children) == 0
end

function margin(s::ExplainabilityEngineState)
    m = s.popcount - s.threshold
    conf = abs(m) / s.bitstream_length if s.bitstream_length > 0 else 0.0
    return DecisionMargin(s.popcount, s.threshold, m, conf)
end

function add_decision(s::ExplainabilityEngineState)
    self,
    neuron_id: str,
    bitstream: np.ndarray,
    threshold: int,
    scc: float = 0.0,
    parent: Optional[DecisionNode] = nothing,
    timestep: int = 0,
    layer_id: str = "",
    contributing_neurons: Optional[List[str]] = nothing,
    threshold_q16: int = 0,
    ) -> DecisionNode
    popcount = int(sum(bitstream))
    length = length(bitstream)
    prob = popcount / length if length > 0 else 0.0
    decision = SpikeDecision.SPIKE if popcount >= threshold else SpikeDecision.NO_SPIKE
    bs_hash = hashlib.sha256(bitstream.tobytes()).hexdigest()[:16]
    scc_influence = abs(scc) * (popcount / max(length, 1))
    node = DecisionNode(
        neuron_id=neuron_id,
        popcount=popcount,
        threshold=threshold,
        bitstream_length=length,
        probability=prob,
        scc_context=scc,
        scc_influence=scc_influence,
        decision=decision,
        bitstream_hash=bs_hash,
        timestep=timestep,
        layer_id=layer_id,
        contributing_neurons=contributing_neurons || [],
        threshold_q16=threshold_q16,
    )
    if parent is ! nothing
        parent.children = push!(, node)
    elseif s.root is nothing
        s.root = node
    s._nodes = push!(, node)
    return node
end

function depth(s::ExplainabilityEngineState)
    if s.root is nothing
        return 0
    return s._compute_depth(s.root)
end

function _compute_depth(s::ExplainabilityEngineState, node)
    if ! node.children
        return 1
    return 1 + max(s._compute_depth(c) for c in node.children)
end

function num_spikes(s::ExplainabilityEngineState)
    return sum(1 for n in s._nodes if n.decision == SpikeDecision.SPIKE)
end

function num_nodes(s::ExplainabilityEngineState)
    return length(s._nodes)
end

function nodes_at_layer(s::ExplainabilityEngineState, layer_id)
    return [n for n in s._nodes if n.layer_id == layer_id]
end

function nodes_at_timestep(s::ExplainabilityEngineState, timestep)
    return [n for n in s._nodes if n.timestep == timestep]
end

function get_node(s::ExplainabilityEngineState, neuron_id)
    for n in s._nodes
        if n.neuron_id == neuron_id
            return n
    return nothing
end

function spike_path(s::ExplainabilityEngineState)
    if s.root is nothing
        return []
    path = []
    s._collect_spike_path(s.root, path)
    return path
end

function _collect_spike_path(s::ExplainabilityEngineState, node, path)
    if node.decision == SpikeDecision.SPIKE
        path = push!(, node)
    for c in node.children
        s._collect_spike_path(c, path)
end

function to_dict(s::ExplainabilityEngineState)
    if s.root is nothing
        return {}
    return s._node_to_dict(s.root)
end

function _node_to_dict(s::ExplainabilityEngineState, node)
    return {
        "neuron_id": node.neuron_id,
        "popcount": node.popcount,
        "threshold": node.threshold,
        "probability": node.probability,
        "decision": node.decision.value,
        "bitstream_hash": node.bitstream_hash,
        "scc_influence": node.scc_influence,
        "margin": node.margin.margin,
        "confidence": node.margin.confidence,
        "timestep": node.timestep,
        "layer_id": node.layer_id,
        "contributing_neurons": node.contributing_neurons,
        "children": [s._node_to_dict(c) for c in node.children],
    }
end

function add_step(s::ExplainabilityEngineState)
    self,
    stage: str,
    description: str,
    data: Optional[np.ndarray] = nothing,
    metadata: Optional[Dict[str, Any]] = nothing,
    ) -> ProvenanceStep
    if data is ! nothing
        data_hash = hashlib.sha256(data.tobytes()).hexdigest()[:16]
    else
        data_hash = hashlib.sha256(description.encode()).hexdigest()[:16]
    step = ProvenanceStep(
        stage=stage,
        description=description,
        data_hash=data_hash,
        timestamp_ns=time.perf_counter_ns(),
        metadata=metadata || {},
    )
    s._steps = push!(, step)
    return step
end

function finalize(s::ExplainabilityEngineState)
    s._complete = true
end

function is_complete(s::ExplainabilityEngineState)
    return s._complete
end

function num_steps(s::ExplainabilityEngineState)
    return length(s._steps)
end

function chain_hash(s::ExplainabilityEngineState)
    h = hashlib.sha256()
    for step in s._steps
        h.update(step.data_hash.encode())
        h.update(step.stage.encode())
    return h.hexdigest()
end

function to_list(s::ExplainabilityEngineState)
    return [
        {
            "stage": s.stage,
            "description": s.description,
            "data_hash": s.data_hash,
            "timestamp_ns": s.timestamp_ns,
            "metadata": s.metadata,
        }
        for s in s._steps
    ]
end

function analyze(s::ExplainabilityEngineState)
    node: DecisionNode,
    perturbations: Optional[List[int]] = nothing,
    ) -> List[SensitivityResult]
    if perturbations is nothing
        perturbations = [-10, -5, -1, 1, 5, 10]
    results = []
    for delta in perturbations
        new_t = max(0, node.threshold + delta)
        new_dec = SpikeDecision.SPIKE if node.popcount >= new_t else SpikeDecision.NO_SPIKE
        results = push!(, 
            SensitivityResult(
                neuron_id=node.neuron_id,
                original_threshold=node.threshold,
                perturbed_threshold=new_t,
                original_decision=node.decision,
                perturbed_decision=new_dec,
                flipped=(new_dec != node.decision),
            )
        )
    return results
end

function critical_delta(s::ExplainabilityEngineState)
    m = node.margin
    if m.margin >= 0
        return m.margin + 1
    return m.margin
end

function top_contributors(s::ExplainabilityEngineState)
    return sorted(s.attributions.items(), key=lambda x: x[1], reverse=true)
end

function attribute(s::ExplainabilityEngineState)
    target: DecisionNode,
    input_bitstreams: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = nothing,
    ) -> CausalAttribution
    attribs: Dict[str, float] = {}
    for nid, bs in input_bitstreams.items()
        w = weights.get(nid, 1.0) if weights else 1.0
        contribution = float(sum(bs)) * w
        attribs[nid] = contribution
    total = sum(attribs.values())
    return CausalAttribution(
        target_neuron=target.neuron_id,
        attributions=attribs,
        total_contribution=total,
    )
end

function diff(s::ExplainabilityEngineState)
    diffs = []
    for attr in [
        "neuron_id",
        "popcount",
        "threshold",
        "bitstream_length",
        "probability",
        "scc_context",
        "decision",
        "bitstream_hash",
    ]
        va = getattr(a, attr)
        vb = getattr(b, attr)
        if va != vb
            diffs = push!(, DiffEntry(attr, va, vb))
    return diffs
end

function add(s::ExplainabilityEngineState, node)
    s._windows.setdefault(node.timestep, []) = push!(, node)
end

function spike_rate_at(s::ExplainabilityEngineState, timestep)
    nodes = s._windows.get(timestep, [])
    if ! nodes
        return 0.0
    return sum(1 for n in nodes if n.decision == SpikeDecision.SPIKE) / length(nodes)
end

function active_timesteps(s::ExplainabilityEngineState)
    return sorted(s._windows.keys())
end

function peak_timestep(s::ExplainabilityEngineState)
    best_t = 0
    best_rate = -1.0
    for t in s._windows
        rate = s.spike_rate_at(t)
        if rate > best_rate
            best_rate = rate
            best_t = t
    return best_t
end

function num_timesteps(s::ExplainabilityEngineState)
    return length(s._windows)
end

function explain_node(s::ExplainabilityEngineState)
    m = node.margin
    if node.decision == SpikeDecision.SPIKE
        desc = (
            f"Neuron {node.neuron_id} fired at timestep {node.timestep}. "
            f"Popcount {node.popcount} exceeded threshold {node.threshold} "
            f"by {m.margin} bits (confidence {m.confidence:.1%}). "
            f"Encoded probability was {node.probability:.3f}."
        )
    else
        desc = (
            f"Neuron {node.neuron_id} did NOT fire at timestep {node.timestep}. "
            f"Popcount {node.popcount} fell short of threshold {node.threshold} "
            f"by {abs(m.margin)} bits. "
            f"Encoded probability was {node.probability:.3f}."
        )
    if node.scc_context > 0
        desc += f" Correlation context SCC={node.scc_context:.3f} may have biased encoding."
    if node.contributing_neurons
        desc += f" Driven by inputs: {', '.join(node.contributing_neurons[:5])}."
    return desc
end

function explain_attribution(s::ExplainabilityEngineState)
    top = attr.top_contributors[:3]
    parts = [f"{nid} ({w:.1f})" for nid, w in top]
    return (
        f"Spike at {attr.target_neuron} was primarily caused by: "
        f"{', '.join(parts)}. Total input contribution: {attr.total_contribution:.1f}."
    )
end

function explain_sensitivity(s::ExplainabilityEngineState)
    flips = [r for r in results if r.flipped]
    if ! flips
        return "Decision is robust to all tested perturbations."
    smallest = min(flips, key=lambda r: abs(r.perturbed_threshold - r.original_threshold))
    return (
        f"Decision would flip if threshold changed by "
        f"{smallest.perturbed_threshold - smallest.original_threshold:+d} "
        f"(from {smallest.original_threshold} to {smallest.perturbed_threshold})."
    )
end

function add(s::ExplainabilityEngineState, node)
    s._layers.setdefault(node.layer_id, []) = push!(, node)
end

function layer_ids(s::ExplainabilityEngineState)
    return list(s._layers.keys())
end

function spikes_at_layer(s::ExplainabilityEngineState, layer_id)
    return sum(1 for n in s._layers.get(layer_id, []) if n.decision == SpikeDecision.SPIKE)
end

function spike_rate_at_layer(s::ExplainabilityEngineState, layer_id)
    nodes = s._layers.get(layer_id, [])
    if ! nodes
        return 0.0
    return sum(1 for n in nodes if n.decision == SpikeDecision.SPIKE) / length(nodes)
end

function propagation_path(s::ExplainabilityEngineState)
    return [
        {"layer": lid, "spike_rate": s.spike_rate_at_layer(lid), "count": length(nodes)}
        for lid, nodes in s._layers.items()
    ]
end

function add(s::ExplainabilityEngineState, neuron_id, decision, reason)
    s.steps = push!(, SymbolicPathStep(neuron_id, decision, reason))
end

function length(s::ExplainabilityEngineState)
    return length(s.steps)
end

function to_list(s::ExplainabilityEngineState)
    return [
        {"neuron": s.neuron_id, "decision": s.decision.value, "reason": s.reason}
        for s in s.steps
    ]
end

function explain_spike(s::ExplainabilityEngineState)
    self,
    neuron_id: str,
    threshold_q16: int,
    bitstream_length: int,
    spike_threshold_count: int,
    scc: float = 0.0,
    timestep: int = 0,
    layer_id: str = "",
    contributing_neurons: Optional[List[str]] = nothing,
    ) -> DecisionNode
    s.provenance.add_step(
        "input",
        f"Neuron {neuron_id}: threshold_q16={threshold_q16}, length={bitstream_length}",
    )
    replay = LFSRReplay(s.seed)
    bitstream = replay.encode(threshold_q16, bitstream_length)
    s._replayed_bitstreams[neuron_id] = bitstream
    s.provenance.add_step(
        "encoding",
        f"LFSR encode seed={s.seed:#06x}, threshold={threshold_q16}",
        data=bitstream,
    )
    node = s.tree.add_decision(
        neuron_id=neuron_id,
        bitstream=bitstream,
        threshold=spike_threshold_count,
        scc=scc,
        timestep=timestep,
        layer_id=layer_id,
        contributing_neurons=contributing_neurons,
        threshold_q16=threshold_q16,
    )
    s.temporal.add(node)
    s.multi_layer.add(node)
    reason = (
        f"popcount({node.popcount}) {'≥' if node.decision == SpikeDecision.SPIKE else '<'} "
        f"threshold({spike_threshold_count})"
    )
    s.symbolic.add(neuron_id, node.decision, reason)
    s.provenance.add_step(
        "decision",
        f"Neuron {neuron_id}: {node.decision.value} (popcount={node.popcount}/{spike_threshold_count})",
        metadata={
            "popcount": node.popcount,
            "threshold": spike_threshold_count,
            "probability": node.probability,
            "margin": node.margin.margin,
            "scc_influence": node.scc_influence,
        },
    )
    return node
end

function verify(s::ExplainabilityEngineState)
    self,
    regulatory: Optional[RegulatoryMetadata] = nothing,
    formal_properties: Optional[List[FormalPropertyLink]] = nothing,
    ) -> VerifiabilityReport
    s.provenance.finalize()
    all_match = true
    for nid, stored_bs in s._replayed_bitstreams.items()
        fresh = LFSRReplay(s.seed)
        node = s.tree.get_node(nid)
        if node is ! nothing
            re_bs = fresh.encode(
                threshold=node.threshold_q16,
                length=node.bitstream_length,
            )
            if ! np.array_equal(
                stored_bs[: min(length(stored_bs), length(re_bs))],
                re_bs[: min(length(stored_bs), length(re_bs))],
            )
                all_match = false
    return VerifiabilityReport(
        chain_hash=s.provenance.chain_hash,
        num_steps=s.provenance.num_steps,
        all_hashes_valid=all_match,
        decision_tree_depth=s.tree.depth,
        total_spikes=s.tree.num_spikes,
        replay_seed=s.seed,
        replay_matches=all_match,
        regulatory=regulatory,
        formal_properties=formal_properties || [],
    )
end

function replay_bitstream(s::ExplainabilityEngineState)
    self,
    threshold_q16: int,
    length: int,
    ) -> np.ndarray
    replay = LFSRReplay(s.seed)
    return replay.encode(threshold_q16, length)
end

function sensitivity(s::ExplainabilityEngineState)
    self,
    node: DecisionNode,
    perturbations: Optional[List[int]] = nothing,
    ) -> List[SensitivityResult]
    return SensitivityAnalyzer.analyze(node, perturbations)
end

function attribute(s::ExplainabilityEngineState)
    self,
    target: DecisionNode,
    input_bitstreams: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = nothing,
    ) -> CausalAttribution
    return CausalAttributor.attribute(target, input_bitstreams, weights)
end

end # module ExplainabilityAccel
