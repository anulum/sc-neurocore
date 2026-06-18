# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neural population decoding algorithms

"""Neural population decoding algorithms."""

from __future__ import annotations

from typing import Any
import numpy as np


def population_vector_decode(
    trains: list[np.ndarray[Any, Any]],
    preferred_directions: np.ndarray[Any, Any],
    window: int = 50,
) -> np.ndarray[Any, Any]:
    """Georgopoulos population vector decoding.

    Each neuron i has a preferred direction (angle in radians).
    Decoded direction per time bin = weighted sum of preferred directions.
    Returns decoded angles per time bin.
    """
    if not trains:
        return np.array([])
    min_len = min(t.size for t in trains)
    n_bins = min_len // window
    if n_bins == 0:
        return np.array([])
    decoded = np.zeros(n_bins)
    for b in range(n_bins):
        sx, sy = 0.0, 0.0
        for i, t in enumerate(trains):
            count = t[b * window : (b + 1) * window].sum()
            sx += count * np.cos(preferred_directions[i])
            sy += count * np.sin(preferred_directions[i])
        decoded[b] = np.arctan2(sy, sx)
    return decoded


def bayesian_decode(
    spike_counts: np.ndarray[Any, Any],
    tuning_rates: np.ndarray[Any, Any],
    prior: np.ndarray[Any, Any] = None,  # type: ignore[assignment]
) -> int:
    """Bayesian MAP decoder. Dayan & Abbott 2001.

    spike_counts: (n_neurons,) observed counts.
    tuning_rates: (n_stimuli, n_neurons) mean rates per stimulus.
    prior: (n_stimuli,) prior probabilities. Uniform if None.
    Returns: MAP stimulus index.
    """
    n_stim, n_neurons = tuning_rates.shape
    if prior is None:
        prior = np.ones(n_stim) / n_stim
    log_posterior = np.log(prior + 1e-30)
    for s in range(n_stim):
        for j in range(n_neurons):
            lam = max(tuning_rates[s, j], 1e-10)
            log_posterior[s] += spike_counts[j] * np.log(lam) - lam
    return int(np.argmax(log_posterior))


def maximum_likelihood_decode(
    spike_counts: np.ndarray[Any, Any], tuning_rates: np.ndarray[Any, Any]
) -> int:
    """Maximum likelihood stimulus decoder. Dayan & Abbott 2001.

    Poisson likelihood: argmax_s prod_j (lambda_j^{n_j} * exp(-lambda_j) / n_j!).
    """
    return bayesian_decode(spike_counts, tuning_rates, prior=None)  # type: ignore[arg-type]


def linear_discriminant_decode(
    train_data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], test_point: np.ndarray[Any, Any]
) -> int:
    """Fisher linear discriminant decoder. Fisher 1936.

    train_data: (n_samples, n_features). labels: (n_samples,). test_point: (n_features,).
    Returns predicted class label.
    """
    classes = np.unique(labels)
    if len(classes) < 2:
        return int(classes[0]) if len(classes) > 0 else 0
    means = {}
    s_w = np.zeros((train_data.shape[1], train_data.shape[1]))
    for c in classes:
        mask = labels == c
        class_data = train_data[mask]
        means[c] = class_data.mean(axis=0)
        diff = class_data - means[c]
        s_w += diff.T @ diff
    s_w += 1e-8 * np.eye(s_w.shape[0])
    s_w_inv = np.linalg.inv(s_w)
    best_class = classes[0]
    best_score = -np.inf
    overall_mean = train_data.mean(axis=0)
    for c in classes:
        w = s_w_inv @ (means[c] - overall_mean)
        score = w @ test_point
        if score > best_score:
            best_score = score
            best_class = c
    return int(best_class)


def naive_bayes_decode(
    train_data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], test_point: np.ndarray[Any, Any]
) -> int:
    """Gaussian naive Bayes decoder. Mitchell 1997.

    Assumes feature independence. Returns predicted class label.
    """
    classes = np.unique(labels)
    n_total = len(labels)
    best_class = classes[0]
    best_log_p = -np.inf
    for c in classes:
        mask = labels == c
        prior = np.log(mask.sum() / n_total)
        class_data = train_data[mask]
        mu = class_data.mean(axis=0)
        var = class_data.var(axis=0) + 1e-10
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var) + (test_point - mu) ** 2 / var)
        log_p = prior + log_likelihood
        if log_p > best_log_p:
            best_log_p = log_p
            best_class = c
    return int(best_class)
