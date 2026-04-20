# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for spike_gnn/spike_gnn

module SpikeGnnAccel

using Statistics, LinearAlgebra

mutable struct SpikeGNNLayerState
    in_features::Float64
    out_features::Float64
    threshold::Float64
    tau_mem::Float64
    W::Float64
    layer_dims::Float64
    T::Float64
end

function SpikeGNNLayerState()
    SpikeGNNLayerState(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 8.0)
end

function forward(s::SpikeGNNLayerState)
    self,
    node_features: np.ndarray,
    adjacency: np.ndarray,
    T: int = 8,
    ) -> np.ndarray
    N = node_features.shape[0]
    rng = np.random.RandomState(42)
    # Aggregate neighbor features (message passing)
    degree = adjacency.sum(axis=1, keepdims=true)
    degree = clamp(degree, 1, nothing)
    aggregated = (adjacency @ node_features) / degree
    # Project through weight matrix
    projected = aggregated @ s.W.T
    # LIF integration over T timesteps
    s._v = zeros((N, s.out_features))
    spike_counts = zeros((N, s.out_features))
    alpha = exp(-1.0 / s.tau_mem)
    for t in 1:T
        # Rate-code input: spike with probability proportional to projected value
        input_spikes = (rng.random(projected.shape) < clamp(projected, 0, 1)).astype(
            np.float64
        )
        s._v = alpha * s._v + (1 - alpha) * input_spikes
        spikes = (s._v >= s.threshold).astype(np.float64)
        s._v -= spikes * s.threshold
        spike_counts += spikes
    return spike_counts
end

function forward(s::SpikeGNNLayerState, node_features, adjacency)
    h = node_features
    for conv in s.convs
        h = conv.forward(h, adjacency, T=s.T)
        # Normalize spike counts to [0, 1] for next layer
        max_val = h.max()
        if max_val > 0:  # pragma: no cover
            h = h / max_val
    return h
end

function graph_classify(s::SpikeGNNLayerState, node_features, adjacency)
    node_out = s.forward(node_features, adjacency)
    graph_vec = node_out.sum(axis=0)
    return int(argmax(graph_vec))
end

function n_layers(s::SpikeGNNLayerState)
    return length(s.convs)
end

end # module SpikeGnnAccel
