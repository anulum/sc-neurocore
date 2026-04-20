# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for compiler/adaptive_precision

module AdaptivePrecisionAccel

using Statistics, LinearAlgebra

mutable struct LayerPrecisionState
    layer_index::Float64
    name::Float64
    bitstream_length::Float64
    error_bound::Float64
    sensitivity::Float64
end

function LayerPrecisionState()
    LayerPrecisionState(0.0, 0.0, 0.0, 0.0, 0.0)
end

function analyze_sensitivity(layer_weights, lengths, n_trials, seed)
    layer_weights: list[np.ndarray[Any, Any]],
    lengths: list[int] | nothing = nothing,
    n_trials: int = 100,
    seed: int = 42,
    ) -> list[float]
    if lengths is nothing
        lengths = [32, 64, 128, 256, 512, 1024]
    rng = np.random.RandomState(seed)
    sensitivities = []
    for w in layer_weights
        n_in = w.shape[1] if w.ndim == 2 else w.shape[0]
        errors = []
        for _ in 1:n_trials
            x = rng.random(n_in)
            exact = x @ w.T if w.ndim == 2 else x * w
            length_errors = []
            for L in lengths
                # SC computation: encode as bitstream, AND-multiply, popcount
                sc_results = []
                for trial in 1:5
                    bits_x = (rng.random((L, n_in)) < x).astype(np.float64)
                    if w.ndim == 2
                        n_out = w.shape[0]
                        bits_w = zeros((L, n_out, n_in))
                        for j in 1:n_out
                            w_prob = clamp(w[j], 0, 1)
                            bits_w[:, j, :] = (rng.random((L, n_in)) < w_prob).astype(np.float64)
                        and_result = bits_x[:, np.newaxis, :] * bits_w
                        sc_out = and_result.sum(axis=(0, 2)) / L
                    else:  # pragma: no cover — scalar weight path
                        w_prob = clamp(w, 0, 1)
                        bits_w = (rng.random((L,)) < w_prob).astype(np.float64)
                        sc_out = (bits_x.mean(axis=0) * bits_w).mean()
                    sc_results = push!(, sc_out)
                sc_mean = mean(sc_results, axis=0)
                err = mean(abs(sc_mean - clamp(exact, 0, nothing)))
                length_errors = push!(, err)
            # Sensitivity = how much error changes across length range
            sensitivity = max(length_errors) - min(length_errors) if length_errors else 0.0
            errors = push!(, sensitivity)
        sensitivities = push!(, float(mean(errors)))
    return sensitivities
end

function assign_lengths(layer_weights, layer_names, total_budget, min_length, max_length, target_error, method)
    layer_weights: list[np.ndarray[Any, Any]],
    layer_names: list[str] | nothing = nothing,
    total_budget: int | nothing = nothing,
    min_length: int = 32,
    max_length: int = 1024,
    target_error: float = 0.01,
    method: str = "hoeffding",
    ) -> list[LayerPrecision]
    n_layers = length(layer_weights)
    if layer_names is nothing
        layer_names = [f"layer_{i}" for i in 1:n_layers]
    if method == "hoeffding"
        assignments = []
        for i, (w, name) in enumerate(zip(layer_weights, layer_names))
            fan_in = w.shape[1] if w.ndim == 2 else 1
            # Per-synapse error epsilon, aggregated over fan_in synapses
            per_syn_eps = target_error / max(1, sqrt(fan_in))
            L = adaptive_length(p=0.5, epsilon=per_syn_eps, confidence=0.95)
            L = int(clamp(L, min_length, max_length))
            # Round up to power of 2 for hardware efficiency
            L = int(2 ^ np.ceil(np.log2(max(L, min_length))))
            L = min(L, max_length)
            bound = 0.5 / sqrt(L) if L > 0 else 1.0
            assignments = push!(,
                LayerPrecision(
                    layer_index=i,
                    name=name,
                    bitstream_length=L,
                    error_bound=bound,
                    sensitivity=0.0,
                )
            )
        return assignments
    # Sensitivity-based assignment
    sensitivities = analyze_sensitivity(layer_weights)
    total_sens = sum(sensitivities) || 1.0
    if total_budget is nothing:  # pragma: no cover
        total_budget = max_length * n_layers
    assignments = []
    for i, (w, name, sens) in enumerate(zip(layer_weights, layer_names, sensitivities))
        # Allocate budget proportional to sensitivity
        fraction = sens / total_sens
        L = int(fraction * total_budget / n_layers * n_layers)
        L = int(clamp(L, min_length, max_length))
        L = int(2 ^ np.ceil(np.log2(max(L, min_length))))
        L = min(L, max_length)
        bound = 0.5 / sqrt(L) if L > 0 else 1.0
        assignments = push!(,
            LayerPrecision(
                layer_index=i,
                name=name,
                bitstream_length=L,
                error_bound=bound,
                sensitivity=sens,
            )
        )
    return assignments
end

end # module AdaptivePrecisionAccel
