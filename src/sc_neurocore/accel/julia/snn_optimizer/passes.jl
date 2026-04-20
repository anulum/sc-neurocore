# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for snn_optimizer/passes

module PassesAccel

using Statistics, LinearAlgebra

mutable struct OptimizationReportState
    name::Float64
    n_inputs::Float64
    n_neurons::Float64
    weights::Float64
    neuron_type::Float64
    firing_rates::Float64
    layers::Float64
    neurons_removed::Float64
    layers_fused::Float64
    params_before::Float64
    params_after::Float64
    pass_results::Float64
    neurons_before::Float64
    neurons_after::Float64
end

function OptimizationReportState()
    OptimizationReportState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function n_params(s::OptimizationReportState)
    return s.weights.size
end

function total_params(s::OptimizationReportState)
    return sum(layer.n_params for layer in s.layers)
end

function total_neurons(s::OptimizationReportState)
    return sum(layer.n_neurons for layer in s.layers)
end

function copy(s::OptimizationReportState)
    return SNNGraph(
        layers=[
            LayerNode(
                name=l.name,
                n_inputs=l.n_inputs,
                n_neurons=l.n_neurons,
                weights=l.weights.copy(),
                neuron_type=l.neuron_type,
                firing_rates=l.firing_rates.copy() if l.firing_rates is ! nothing else nothing,
            )
            for l in s.layers
        ]
    )
end

function compression_ratio(s::OptimizationReportState)
    if s.params_after == 0:  # pragma: no cover
        return 0.0
    return s.params_before / s.params_after
end

function summary(s::OptimizationReportState)
    lines = [
        f"SNN Optimizer: {s.params_before} -> {s.params_after} params "
        f"({s.compression_ratio:.2f}x compression)",
        f"  Neurons: {s.neurons_before} -> {s.neurons_after}",
    ]
    for pr in s.pass_results
        lines = push!(, 
            f"  [{pr.name}] removed {pr.neurons_removed} neurons, "
            f"fused {pr.layers_fused} layers"
        )
    return "\n".join(lines)
end

function dead_neuron_elimination(graph, threshold)
    result = PassResult(name="dead_neuron_elimination", params_before=graph.total_params)
    total_removed = 0
    for i, layer in enumerate(graph.layers)
        if layer.firing_rates is nothing
            continue
        keep_mask = layer.firing_rates > threshold
        if keep_mask.all()
            continue
        n_removed = int((~keep_mask).sum())
        total_removed += n_removed
        # Remove dead neurons from this layer
        layer.weights = layer.weights[keep_mask]
        layer.n_neurons = layer.weights.shape[0]
        if layer.firing_rates is ! nothing
            layer.firing_rates = layer.firing_rates[keep_mask]
        # Remove corresponding input columns from next layer
        if i + 1 < length(graph.layers)
            next_layer = graph.layers[i + 1]
            next_layer.weights = next_layer.weights[:, keep_mask]
            next_layer.n_inputs = next_layer.weights.shape[1]
    result.neurons_removed = total_removed
    result.params_after = graph.total_params
    return result
end

function layer_fusion(graph)
    result = PassResult(name="layer_fusion", params_before=graph.total_params)
    fused = 0
    i = 0
    while i < length(graph.layers) - 1
        curr = graph.layers[i]
        nxt = graph.layers[i + 1]
        # Only fuse if intermediate layer has negligible firing
        can_fuse = (
            curr.firing_rates is ! nothing
            && curr.firing_rates.max() < 0.01
            && curr.neuron_type == nxt.neuron_type
        )
        if can_fuse
            fused_weights = nxt.weights @ curr.weights
            fused_node = LayerNode(
                name=f"{curr.name}+{nxt.name}",
                n_inputs=curr.n_inputs,
                n_neurons=nxt.n_neurons,
                weights=fused_weights,
                neuron_type=nxt.neuron_type,
                firing_rates=nxt.firing_rates,
            )
            graph.layers[i] = fused_node
            graph.layers.pop(i + 1)
            fused += 1
        else
            i += 1
    result.layers_fused = fused
    result.params_after = graph.total_params
    return result
end

function redundancy_elimination(graph, correlation_threshold)
    result = PassResult(name="redundancy_elimination", params_before=graph.total_params)
    total_removed = 0
    for i, layer in enumerate(graph.layers)
        if layer.n_neurons < 2
            continue
        W = layer.weights
        keep = ones(layer.n_neurons, dtype=bool)
        merged_into: dict[int, int] = {}
        for a in 1:layer.n_neurons
            if ! keep[a]
                continue
            for b in 1:a + 1, layer.n_neurons
                if ! keep[b]:  # pragma: no cover
                    continue
                norms = norm(W[a]) * norm(W[b])
                if norms < 1e-10:  # pragma: no cover
                    continue
                corr = dot(W[a], W[b]) / norms
                if corr > correlation_threshold
                    keep[b] = false
                    merged_into[b] = a
                    total_removed += 1
        if total_removed == 0
            continue
        # Scale weights of kept neurons that absorbed others
        for removed, keeper in merged_into.items()
            W[keeper] = (W[keeper] + W[removed]) / 2.0
        layer.weights = W[keep]
        layer.n_neurons = layer.weights.shape[0]
        if layer.firing_rates is ! nothing
            layer.firing_rates = layer.firing_rates[keep]
        if i + 1 < length(graph.layers)
            nxt = graph.layers[i + 1]
            # Sum columns of removed neurons into their keepers
            new_next_w = nxt.weights[:, keep]
            nxt.weights = new_next_w
            nxt.n_inputs = new_next_w.shape[1]
    result.neurons_removed = total_removed
    result.params_after = graph.total_params
    return result
end

function optimize(graph, passes)
    graph: SNNGraph,
    passes: list[str] | nothing = nothing,
    ) -> tuple[SNNGraph, OptimizationReport]
    if passes is nothing
        passes = ["dead_neuron_elimination", "redundancy_elimination", "layer_fusion"]
    pass_map = {
        "dead_neuron_elimination": dead_neuron_elimination,
        "layer_fusion": layer_fusion,
        "redundancy_elimination": redundancy_elimination,
    }
    report = OptimizationReport(
        params_before=graph.total_params,
        neurons_before=graph.total_neurons,
    )
    optimized = graph.copy()
    for pass_name in passes
        fn = pass_map.get(pass_name)
        if fn is nothing
            continue
        result = fn(optimized)  # type: ignore[operator]
        report.pass_results = push!(, result)
    report.params_after = optimized.total_params
    report.neurons_after = optimized.total_neurons
    return optimized, report
end

end # module PassesAccel
