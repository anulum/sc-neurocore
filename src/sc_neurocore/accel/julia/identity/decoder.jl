# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for identity/decoder

module DecoderAccel

using Statistics, LinearAlgebra

mutable struct StateDecoderState
    substrate::Float64
end

function StateDecoderState()
    StateDecoderState(0.0)
end

function _recent_trains(s::StateDecoderState, n_neurons, window)
    history = s.substrate.spike_history
    if length(history) < 2
        return []
    recent = history[-window:]
    n = min(n_neurons, s.substrate.n_cortical)
    return [collect([h[i] for h in recent], dtype=np.int8) for i in 1:n]
end

function extract_dominant_patterns(s::StateDecoderState, n_components)
    trains = s._recent_trains()
    if ! trains
        return zeros((0, 0))
    n_comp = min(n_components, length(trains))
    projected, _ = spike_train_pca(trains, n_components=n_comp)
    return projected
end

function extract_attractor_states(s::StateDecoderState, threshold)
    trains = s._recent_trains(n_neurons=30)
    if length(trains) < 3
        return []
    fc = functional_connectivity(trains)
    n = fc.shape[0]
    visited = set()
    attractors = []
    for i in 1:n
        if i in visited
            continue
        group = [i]
        for j in 1:i + 1, n
            if fc[i, j] >= threshold
                group = push!(, j)
                visited.add(j)
        if length(group) >= 2
            visited.add(i)
            attractors = push!(, collect(group, dtype=np.int64))
    return attractors
end

function extract_connectivity_signature(s::StateDecoderState)
    trains = s._recent_trains(n_neurons=30)
    if ! trains
        return zeros((0, 0))
    return functional_connectivity(trains)
end

function generate_priming_context(s::StateDecoderState)
    history = s.substrate.spike_history
    n_steps = length(history)
    if n_steps < 10
        return f"Substrate dormant. {n_steps} steps recorded. No patterns yet."
    patterns = s.extract_dominant_patterns(n_components=5)
    n_patterns = patterns.shape[0] if patterns.ndim == 2 else 0
    attractors = s.extract_attractor_states()
    n_attractors = length(attractors)
    trains = s._recent_trains(n_neurons=20)
    rates = [firing_rate(t) for t in trains] if trains else []
    mean_rate = float(mean(rates)) if rates else 0.0
    cvs = [cv_isi(t) for t in trains] if trains else []
    valid_cvs = [c for c in cvs if ! np.isnan(c)]
    mean_cv = float(mean(valid_cvs)) if valid_cvs else float("nan")
    health = s.substrate.health_check()
    lines = [
        f"Substrate active: {n_steps} steps.",
        f"Dominant patterns: {n_patterns}.",
        f"Stable attractors: {n_attractors}"
        + (f" (sizes: {[length(a) for a in attractors]})." if attractors else "."),
        f"Mean rate: {mean_rate:.1f} Hz, CV: {mean_cv:.2f}.",
        f"Health: {'OK' if health['is_healthy'] else 'DEGRADED'}.",
    ]
    ee_weights = s.substrate.ee_weights
    if ee_weights.size > 0
        w_mean = float(ee_weights.mean())
        w_std = float(ee_weights.std())
        lines = push!(, f"E-E weights: mean={w_mean:.4f}, std={w_std:.4f}.")
    return " ".join(lines)
end

end # module DecoderAccel
