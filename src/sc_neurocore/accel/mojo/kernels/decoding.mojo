# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for decoding

fn population_vector_decode(trains: Int, preferred_directions: Int, window: Int) -> Int:
    var _population_vector_decode_line = 'trains: list[ndarray],'
    var _population_vector_decode_line = 'preferred_directions: ndarray,'
    var _population_vector_decode_line = 'window: int = 50,'
    var _population_vector_decode_line = ') -> ndarray:'
    var _population_vector_decode_line = 'if not trains:'
    return 0  # return array([])
    var _population_vector_decode_line = 'min_len = min(t.size for t in trains)'
    var _population_vector_decode_line = 'n_bins = min_len // window'
    var _population_vector_decode_line = 'if n_bins == 0:'
    return 0  # return array([])
    var _population_vector_decode_line = 'decoded = zeros(n_bins)'
    var _population_vector_decode_line = 'for b in range(n_bins):'
    var _population_vector_decode_line = 'sx, sy = 0.0, 0.0'
    var _population_vector_decode_line = 'for i, t in enumerate(trains):'
    var _population_vector_decode_line = 'count = t[b * window : (b + 1) * window].sum()'
    var _population_vector_decode_line = 'sx += count * cos(preferred_directions[i])'
    var _population_vector_decode_line = 'sy += count * sin(preferred_directions[i])'
    var _population_vector_decode_line = 'decoded[b] = arctan2(sy, sx)'
    return 0  # return decoded

fn bayesian_decode(spike_counts: Int, tuning_rates: Int, prior: Int) -> Int:
    var _bayesian_decode_line = 'spike_counts: ndarray,'
    var _bayesian_decode_line = 'tuning_rates: ndarray,'
    var _bayesian_decode_line = 'prior: ndarray = 0,  # type: ignore[assignment]'
    var _bayesian_decode_line = ') -> int:'
    var _bayesian_decode_line = 'n_stim, n_neurons = tuning_rates.shape'
    var _bayesian_decode_line = 'if prior is 0:'
    var _bayesian_decode_line = 'prior = ones(n_stim) / n_stim'
    var _bayesian_decode_line = 'log_posterior = log(prior + 1e-30)'
    var _bayesian_decode_line = 'for s in range(n_stim):'
    var _bayesian_decode_line = 'for j in range(n_neurons):'
    var _bayesian_decode_line = 'lam = max(tuning_rates[s, j], 1e-10)'
    var _bayesian_decode_line = 'log_posterior[s] += spike_counts[j] * log(lam) - lam'
    return 0  # return int(argmax(log_posterior))

fn maximum_likelihood_decode(spike_counts: Int, tuning_rates: Int) -> Int:
    return 0  # return bayesian_decode(spike_counts, tuning_rates,

fn linear_discriminant_decode(train_data: Int, labels: Int, test_point: Int) -> Int:
    var _linear_discriminant_decode_line = 'train_data: ndarray, labels: ndarray, test_point: ndarray'
    var _linear_discriminant_decode_line = ') -> int:'
    var _linear_discriminant_decode_line = 'classes = unique(labels)'
    var _linear_discriminant_decode_line = 'if len(classes) < 2:'
    return 0  # return int(classes[0]) if len(classes) > 0 else 0
    var _linear_discriminant_decode_line = 'means = {}'
    var _linear_discriminant_decode_line = 's_w = zeros((train_data.shape[1], train_data.shape[1]))'
    var _linear_discriminant_decode_line = 'for c in classes:'
    var _linear_discriminant_decode_line = 'mask = labels == c'
    var _linear_discriminant_decode_line = 'class_data = train_data[mask]'
    var _linear_discriminant_decode_line = 'means[c] = class_data.mean(axis=0)'
    var _linear_discriminant_decode_line = 'diff = class_data - means[c]'
    var _linear_discriminant_decode_line = 's_w += diff.T @ diff'
    var _linear_discriminant_decode_line = 's_w += 1e-8 * eye(s_w.shape[0])'
    var _linear_discriminant_decode_line = 's_w_inv = linalg.inv(s_w)'
    var _linear_discriminant_decode_line = 'best_class = classes[0]'
    var _linear_discriminant_decode_line = 'best_score = -inf'
    var _linear_discriminant_decode_line = 'overall_mean = train_data.mean(axis=0)'
    var _linear_discriminant_decode_line = 'for c in classes:'
    var _linear_discriminant_decode_line = 'w = s_w_inv @ (means[c] - overall_mean)'
    var _linear_discriminant_decode_line = 'score = w @ test_point'
    var _linear_discriminant_decode_line = 'if score > best_score:'
    var _linear_discriminant_decode_line = 'best_score = score'
    var _linear_discriminant_decode_line = 'best_class = c'
    return 0  # return int(best_class)

fn naive_bayes_decode(train_data: Int, labels: Int, test_point: Int) -> Int:
    var _naive_bayes_decode_line = 'classes = unique(labels)'
    var _naive_bayes_decode_line = 'n_total = len(labels)'
    var _naive_bayes_decode_line = 'best_class = classes[0]'
    var _naive_bayes_decode_line = 'best_log_p = -inf'
    var _naive_bayes_decode_line = 'for c in classes:'
    var _naive_bayes_decode_line = 'mask = labels == c'
    var _naive_bayes_decode_line = 'prior = log(mask.sum() / n_total)'
    var _naive_bayes_decode_line = 'class_data = train_data[mask]'
    var _naive_bayes_decode_line = 'mu = class_data.mean(axis=0)'
    var _naive_bayes_decode_line = 'var = class_data.var(axis=0) + 1e-10'
    var _naive_bayes_decode_line = 'log_likelihood = -0.5 * sum(log(2 * pi * var) + (test_point '
    var _naive_bayes_decode_line = 'log_p = prior + log_likelihood'
    var _naive_bayes_decode_line = 'if log_p > best_log_p:'
    var _naive_bayes_decode_line = 'best_log_p = log_p'
    var _naive_bayes_decode_line = 'best_class = c'
    return 0  # return int(best_class)
