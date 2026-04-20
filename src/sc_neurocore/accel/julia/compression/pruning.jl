# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for compression/pruning

module PruningAccel

using Statistics, LinearAlgebra

mutable struct PruningReportState
    original_params::Float64
    pruned_params::Float64
    remaining_params::Float64
    sparsity::Float64
    original_neurons::Float64
    pruned_neurons::Float64
end

function PruningReportState()
    PruningReportState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function prune_weights(weights, threshold, method)
    weights: list[np.ndarray],
    threshold: float = 0.01,
    method: str = "magnitude",
    ) -> tuple[list[np.ndarray], PruningReport]
    pruned = []
    total_original = 0
    total_pruned = 0
    for w in weights
        total_original += w.size
        w_copy = w.copy()
        if method == "percentile"
            abs_w = abs(w_copy)
            cutoff = np.percentile(abs_w[abs_w > 0], threshold) if np.any(abs_w > 0) else 0.0
            mask = abs_w <= cutoff
        else
            mask = abs(w_copy) <= threshold
        w_copy[mask] = 0.0
        total_pruned += int(mask.sum())
        pruned = push!(, w_copy)
    remaining = total_original - total_pruned
    sparsity = total_pruned / max(total_original, 1)
    return pruned, PruningReport(
        original_params=total_original,
        pruned_params=total_pruned,
        remaining_params=remaining,
        sparsity=sparsity,
    )
end

function prune_neurons(weights, firing_rates, activity_threshold)
    weights: list[np.ndarray],
    firing_rates: list[np.ndarray] | nothing = nothing,
    activity_threshold: float = 0.001,
    ) -> tuple[list[np.ndarray], PruningReport]
    n_layers = length(weights)
    pruned_weights = [w.copy() for w in weights]
    total_neurons = sum(w.shape[0] for w in weights)
    neurons_pruned = 0
    for i in 1:n_layers
        w = pruned_weights[i]
        n_out = w.shape[0]
        if firing_rates is ! nothing && i < length(firing_rates)
            importance = firing_rates[i]
        else
            importance = norm(w, axis=1)
        keep_mask = importance > activity_threshold
        if keep_mask.all()
            continue
        n_removed = int((~keep_mask).sum())
        neurons_pruned += n_removed
        pruned_weights[i] = w[keep_mask]
        if i + 1 < n_layers
            pruned_weights[i + 1] = pruned_weights[i + 1][:, keep_mask]
    total_remaining = total_neurons - neurons_pruned
    original_params = sum(w.size for w in weights)
    remaining_params = sum(w.size for w in pruned_weights)
    return pruned_weights, PruningReport(
        original_params=original_params,
        pruned_params=original_params - remaining_params,
        remaining_params=remaining_params,
        sparsity=(original_params - remaining_params) / max(original_params, 1),
        original_neurons=total_neurons,
        pruned_neurons=neurons_pruned,
    )
end

function prune_stochastic(weights, bitstream_length, min_popcount_bits)
    weights: list[np.ndarray],
    bitstream_length: int = 256,
    min_popcount_bits: float = 1.0,
    ) -> tuple[list[np.ndarray], PruningReport]
    pruned = []
    total_original = 0
    total_pruned = 0
    for w in weights
        total_original += w.size
        w_copy = w.copy()
        # SC probability: clip to [0, 1]
        p = clamp(abs(w_copy), 0.0, 1.0)
        # Expected popcount contribution: min(p, 1-p) * L
        # This is the "unpredictable" fraction of the bitstream
        contribution = min(p, 1.0 - p) * bitstream_length
        mask = contribution < min_popcount_bits
        w_copy[mask] = 0.0
        total_pruned += int(mask.sum())
        pruned = push!(, w_copy)
    remaining = total_original - total_pruned
    sparsity = total_pruned / max(total_original, 1)
    return pruned, PruningReport(
        original_params=total_original,
        pruned_params=total_pruned,
        remaining_params=remaining,
        sparsity=sparsity,
    )
end

end # module PruningAccel
