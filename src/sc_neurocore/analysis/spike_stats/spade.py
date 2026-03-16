# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""SPADE -- Spike Pattern Detection and Evaluation.

Torre, Canova, Denker, Gerstein, Helias, Gruen (2013)
"Statistical evaluation of synchronous spike patterns extracted by
frequent item set mining." Front. Comput. Neurosci. 7:132.

Simplified hash-based implementation for populations up to ~50 neurons.
"""

from __future__ import annotations


import numpy as np


def _find_frequent_itemsets(binary_matrix, min_support, max_size):
    """Apriori-style frequent itemset mining on a binary neuron x time matrix."""
    n_neurons, n_bins = binary_matrix.shape
    neuron_ids = list(range(n_neurons))

    # Level-1: single neurons exceeding support
    freq = []
    candidates_k = []
    for n_id in neuron_ids:
        cnt = int(binary_matrix[n_id].sum())
        if cnt >= min_support:
            s = frozenset([n_id])
            freq.append((s, cnt))
            candidates_k.append(s)

    # Level k >= 2: grow candidates from (k-1) pairs sharing k-2 elements
    for k in range(2, max_size + 1):
        if len(candidates_k) < 2:
            break
        new_candidates = set()
        prev = candidates_k
        for i in range(len(prev)):
            for j in range(i + 1, len(prev)):
                union = prev[i] | prev[j]
                if len(union) == k:
                    new_candidates.add(union)

        candidates_k = []
        for s in new_candidates:
            idx = sorted(s)
            active = binary_matrix[idx[0]]
            for nid in idx[1:]:
                active = active & binary_matrix[nid]
            cnt = int(active.sum())
            if cnt >= min_support:
                freq.append((s, cnt))
                candidates_k.append(s)

    return freq


def _extend_to_spatiotemporal(trains, itemsets, bin_ms, dt, max_lag_bins=10):
    """Extend synchronous itemsets to spatiotemporal patterns with lags."""
    n_neurons = len(trains)
    bin_steps = max(1, int(bin_ms / (dt * 1000)))
    duration = max(t.size for t in trains)
    n_bins = duration // bin_steps

    patterns = []
    for neurons_fs, sync_count in itemsets:
        if len(neurons_fs) < 2:
            continue
        neuron_list = sorted(neurons_fs)
        ref = neuron_list[0]

        # Binary arrays per bin for each neuron
        ref_bins = np.zeros(n_bins, dtype=np.int8)
        for b in range(n_bins):
            start = b * bin_steps
            end = min(start + bin_steps, trains[ref].size)
            if trains[ref][start:end].any():
                ref_bins[b] = 1

        best_lags = {ref: 0}
        best_count = int(ref_bins.sum())
        coincidence = ref_bins.copy()

        for nid in neuron_list[1:]:
            best_lag = 0
            best_overlap = 0
            for lag in range(max_lag_bins + 1):
                nbins_shifted = np.zeros(n_bins, dtype=np.int8)
                for b in range(n_bins):
                    src_b = b - lag
                    if 0 <= src_b < n_bins:
                        start = src_b * bin_steps
                        end = min(start + bin_steps, trains[nid].size)
                        if trains[nid][start:end].any():
                            nbins_shifted[b] = 1
                overlap = int((coincidence & nbins_shifted).sum())
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_lag = lag

            best_lags[nid] = best_lag
            nbins_best = np.zeros(n_bins, dtype=np.int8)
            for b in range(n_bins):
                src_b = b - best_lag
                if 0 <= src_b < n_bins:
                    start = src_b * bin_steps
                    end = min(start + bin_steps, trains[nid].size)
                    if trains[nid][start:end].any():
                        nbins_best[b] = 1
            coincidence = coincidence & nbins_best
            best_count = int(coincidence.sum())

        if best_count > 0:
            patterns.append(
                {
                    "neurons": neuron_list,
                    "lags": [best_lags[n] for n in neuron_list],
                    "count": best_count,
                }
            )

    return patterns


def spade_detect(
    trains,
    bin_ms=5.0,
    dt=0.001,
    min_support=3,
    max_pattern_size=5,
    n_surrogates=100,
    alpha=0.05,
    seed=42,
):
    """Detect repeated spatiotemporal spike patterns with significance testing."""
    n_neurons = len(trains)
    if n_neurons < 2:
        return []

    bin_steps = max(1, int(bin_ms / (dt * 1000)))
    duration = max(t.size for t in trains)
    n_bins = duration // bin_steps
    if n_bins == 0:
        return []

    # Build binary matrix (neurons x time_bins)
    binary_matrix = np.zeros((n_neurons, n_bins), dtype=np.int8)
    for i, t in enumerate(trains):
        for b in range(n_bins):
            start = b * bin_steps
            end = min(start + bin_steps, t.size)
            if t[start:end].any():
                binary_matrix[i, b] = 1

    itemsets = _find_frequent_itemsets(binary_matrix, min_support, max_pattern_size)
    if not itemsets:
        return []

    patterns = _extend_to_spatiotemporal(trains, itemsets, bin_ms, dt, max_lag_bins=10)
    if not patterns:
        return []

    # Significance: compare each pattern count against surrogate distribution
    rng = np.random.default_rng(seed)
    results = []
    for pat in patterns:
        surr_counts = np.zeros(n_surrogates, dtype=np.int32)
        for s in range(n_surrogates):
            surr_trains = []
            for i in range(n_neurons):
                shifted = np.roll(trains[i], rng.integers(-bin_steps * 5, bin_steps * 5 + 1))
                surr_trains.append(shifted)
            surr_binary = np.zeros((n_neurons, n_bins), dtype=np.int8)
            for i, t in enumerate(surr_trains):
                for b in range(n_bins):
                    start = b * bin_steps
                    end = min(start + bin_steps, t.size)
                    if t[start:end].any():
                        surr_binary[i, b] = 1
            # Count coincidences for this pattern's neurons with lags
            neuron_list = pat["neurons"]
            lags = pat["lags"]
            coincidence = np.ones(n_bins, dtype=np.int8)
            for nid, lag in zip(neuron_list, lags):
                nbins_n = np.zeros(n_bins, dtype=np.int8)
                for b in range(n_bins):
                    src_b = b - lag
                    if 0 <= src_b < n_bins:
                        nbins_n[b] = surr_binary[nid, src_b]
                coincidence = coincidence & nbins_n
            surr_counts[s] = coincidence.sum()

        p_value = float((surr_counts >= pat["count"]).sum() + 1) / (n_surrogates + 1)
        if p_value <= alpha:
            results.append(
                {
                    "neurons": pat["neurons"],
                    "lags": pat["lags"],
                    "count": pat["count"],
                    "p_value": p_value,
                }
            )

    return results
