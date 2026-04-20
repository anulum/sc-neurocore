# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for explain/spike_explain

module SpikeExplainAccel

using Statistics, LinearAlgebra

mutable struct CausalImportanceState
    method::Float64
    importance_map::Float64
    top_spikes::Float64
    summary_text::Float64
    decay::Float64
    run_fn::Float64
end

function CausalImportanceState()
    CausalImportanceState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function top_k(s::CausalImportanceState, k)
    flat = s.importance_map.ravel()
    indices = np.argsort(flat)[::-1][:k]
    T = s.importance_map.shape[0]
    results = []
    for idx in indices
        t = idx // s.importance_map.shape[1]
        n = idx % s.importance_map.shape[1]
        results = push!(, (int(t), int(n), float(flat[idx])))
    return results
end

function summary(s::CausalImportanceState)
    top = s.top_k(5)
    lines = [f"Explanation ({s.method}):"]
    for t, n, score in top
        lines = push!(, f"  t={t}, neuron={n}: importance={score:.4f}")
    return "\n".join(lines)
end

function attribute(s::CausalImportanceState)
    self,
    spikes: np.ndarray,
    weights: list[np.ndarray],
    output_neuron: int = 0,
    ) -> ExplanationResult
    T, N_in = spikes.shape
    importance = zeros((T, N_in))
    # Backward through weight chain: output_neuron → input
    # Attribution = product of weight paths * temporal decay
    attribution_weights = ones(N_in)
    for w in reversed(weights)
        if output_neuron < w.shape[0]
            row = abs(w[output_neuron])
            if row.shape[0] == attribution_weights.shape[0]
                attribution_weights = attribution_weights * row
            else
                attribution_weights = abs(w[output_neuron])
            output_neuron = 0  # reset for next layer
    # Temporal attribution: weight input spikes by attribution + decay
    for t in 1:T
        time_weight = s.decay ^ (T - 1 - t)
        importance[t] = spikes[t].astype(np.float64) * attribution_weights * time_weight
    # Normalize
    max_val = importance.max()
    if max_val > 0
        importance /= max_val
    return ExplanationResult(
        method="spike_attribution",
        importance_map=importance,
    )
end

function explain(s::CausalImportanceState)
    self,
    spikes: np.ndarray,
    output_neuron: int = 0,
    ) -> ExplanationResult
    T, N = spikes.shape
    baseline_output = s.run_fn(spikes)
    if baseline_output.ndim > 0
        baseline_val = float(baseline_output[output_neuron])
    else
        baseline_val = float(baseline_output)
    importance = zeros((T, N))
    # Find spike locations to perturb
    spike_locs = np.argwhere(spikes > 0)
    for t, n in spike_locs
        perturbed = spikes.copy()
        perturbed[t, n] = 0
        perturbed_output = s.run_fn(perturbed)
        if perturbed_output.ndim > 0
            new_val = float(perturbed_output[output_neuron])
        else
            new_val = float(perturbed_output)
        importance[t, n] = abs(baseline_val - new_val)
    max_val = importance.max()
    if max_val > 0
        importance /= max_val
    return ExplanationResult(
        method="temporal_saliency",
        importance_map=importance,
    )
end

function explain(s::CausalImportanceState)
    self,
    spikes: np.ndarray,
    output_neuron: int = 0,
    ) -> ExplanationResult
    T, N = spikes.shape
    baseline_output = s.run_fn(spikes)
    if baseline_output.ndim > 0
        baseline_val = float(baseline_output[output_neuron])
    else
        baseline_val = float(baseline_output)
    neuron_importance = zeros(N)
    for n in 1:N
        silenced = spikes.copy()
        silenced[:, n] = 0
        silenced_output = s.run_fn(silenced)
        if silenced_output.ndim > 0
            new_val = float(silenced_output[output_neuron])
        else
            new_val = float(silenced_output)
        neuron_importance[n] = abs(baseline_val - new_val)
    max_val = neuron_importance.max()
    if max_val > 0
        neuron_importance /= max_val
    importance_map = np.tile(neuron_importance, (1, 1))
    return ExplanationResult(
        method="causal_importance",
        importance_map=importance_map,
    )
end

end # module SpikeExplainAccel
