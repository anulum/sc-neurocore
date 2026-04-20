# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/decoding

module DecodingAccel

using Statistics, LinearAlgebra

function population_vector_decode(trains, preferred_directions, window)
    trains: list[np.ndarray],
    preferred_directions: np.ndarray,
    window: int = 50,
    ) -> np.ndarray
    if ! trains
        return collect([])
    min_len = min(t.size for t in trains)
    n_bins = min_len // window
    if n_bins == 0
        return collect([])
    decoded = zeros(n_bins)
    for b in 1:n_bins
        sx, sy = 0.0, 0.0
        for i, t in enumerate(trains)
            count = t[b * window : (b + 1) * window].sum()
            sx += count * cos(preferred_directions[i])
            sy += count * sin(preferred_directions[i])
        decoded[b] = np.arctan2(sy, sx)
    return decoded
end

function bayesian_decode(spike_counts, tuning_rates, prior)
    spike_counts: np.ndarray,
    tuning_rates: np.ndarray,
    prior: np.ndarray = nothing,  # type: ignore[assignment]
    ) -> int
    n_stim, n_neurons = tuning_rates.shape
    if prior is nothing
        prior = ones(n_stim) / n_stim
    log_posterior = log(prior + 1e-30)
    for s in 1:n_stim
        for j in 1:n_neurons
            lam = max(tuning_rates[s, j], 1e-10)
            log_posterior[s] += spike_counts[j] * log(lam) - lam
    return int(argmax(log_posterior))
end

function maximum_likelihood_decode(spike_counts, tuning_rates)
    return bayesian_decode(spike_counts, tuning_rates, prior=nothing)
end

function linear_discriminant_decode(train_data, labels, test_point)
    train_data: np.ndarray, labels: np.ndarray, test_point: np.ndarray
    ) -> int
    classes = np.unique(labels)
    if length(classes) < 2
        return int(classes[0]) if length(classes) > 0 else 0
    means = {}
    s_w = zeros((train_data.shape[1], train_data.shape[1]))
    for c in classes
        mask = labels == c
        class_data = train_data[mask]
        means[c] = class_data.mean(axis=0)
        diff = class_data - means[c]
        s_w += diff.T @ diff
    s_w += 1e-8 * np.eye(s_w.shape[0])
    s_w_inv = np.linalg.inv(s_w)
    best_class = classes[0]
    best_score = -Inf
    overall_mean = train_data.mean(axis=0)
    for c in classes
        w = s_w_inv @ (means[c] - overall_mean)
        score = w @ test_point
        if score > best_score
            best_score = score
            best_class = c
    return int(best_class)
end

function naive_bayes_decode(train_data, labels, test_point)
    classes = np.unique(labels)
    n_total = length(labels)
    best_class = classes[0]
    best_log_p = -Inf
    for c in classes
        mask = labels == c
        prior = log(mask.sum() / n_total)
        class_data = train_data[mask]
        mu = class_data.mean(axis=0)
        var = class_data.var(axis=0) + 1e-10
        log_likelihood = -0.5 * sum(log(2 * pi * var) + (test_point - mu) ^ 2 / var)
        log_p = prior + log_likelihood
        if log_p > best_log_p
            best_log_p = log_p
            best_class = c
    return int(best_class)
end

end # module DecodingAccel
