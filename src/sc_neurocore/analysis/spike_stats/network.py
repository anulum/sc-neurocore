# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network-level spike train analysis: connectivity,

"""Network-level spike train analysis: connectivity, assemblies, synfire chains."""

from __future__ import annotations

from typing import Any
import numpy as np

from .basic import bin_spike_train
from .correlation import cross_correlation


def functional_connectivity(
    trains: list[np.ndarray[Any, Any]], max_lag_ms: float = 20.0, dt: float = 0.001
) -> np.ndarray[Any, Any]:
    """Infer functional connectivity matrix from peak cross-correlation.

    Returns NxN matrix where entry (i,j) is max |cross-correlation|
    between neuron i and neuron j within +/-max_lag.
    """
    n = len(trains)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                mat[i, j] = 1.0
                continue
            cc, _ = cross_correlation(trains[i], trains[j], max_lag_ms=max_lag_ms, dt=dt)
            peak = np.abs(cc).max() if cc.size > 0 else 0.0
            mat[i, j] = mat[j, i] = peak
    return mat


def unitary_events(
    trains: list[np.ndarray[Any, Any]], bin_size: int = 5, alpha: float = 0.05
) -> list[int]:
    """Unitary event analysis. Gruen et al. 2002.

    Detects bins where coincident spikes exceed chance (Poisson assumption).
    Returns list of significant bin indices.
    """
    n_trains = len(trains)
    if n_trains < 2:
        return []
    binned = [bin_spike_train(t, bin_size) for t in trains]
    min_bins = min(b.size for b in binned)
    mat = np.array([b[:min_bins] for b in binned])
    active = (mat > 0).astype(np.float64)
    coincidence = np.prod(active, axis=0)
    rates = active.mean(axis=1)
    expected_rate = np.prod(rates)
    significant_bins = []
    for k in range(min_bins):
        if coincidence[k] > 0:
            p_val = expected_rate**n_trains
            if p_val < alpha:
                significant_bins.append(k)
    return significant_bins


def cell_assembly_detection(
    trains: list[np.ndarray[Any, Any]], bin_size: int = 5, threshold: float = 2.0
) -> list[list[int]]:
    """Cell assembly detection via PCA on binned spike matrix. Lopes-dos-Santos et al. 2013.

    Returns list of assemblies (each a list of neuron indices above threshold).
    """
    n = len(trains)
    if n < 3:
        return []
    binned = [bin_spike_train(t, bin_size).astype(np.float64) for t in trains]
    min_bins = min(b.size for b in binned)
    mat = np.array([b[:min_bins] for b in binned])
    mat -= mat.mean(axis=1, keepdims=True)
    std = mat.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    mat /= std
    corr = mat @ mat.T / min_bins
    eigvals, eigvecs = np.linalg.eigh(corr)
    # Marcenko-Pastur upper bound: lambda_max = (1 + sqrt(n/T))^2
    q = n / min_bins
    mp_upper = (1.0 + np.sqrt(q)) ** 2
    assemblies = []
    for i in range(n):
        if eigvals[i] > mp_upper:
            members = np.where(np.abs(eigvecs[:, i]) > threshold / np.sqrt(n))[0]
            if len(members) >= 2:
                assemblies.append(members.tolist())
    return assemblies


def synfire_chain_detection(
    trains: list[np.ndarray[Any, Any]],
    dt: float = 0.001,
    max_delay_ms: float = 20.0,
    min_chain_length: int = 3,
) -> list[list[int]]:
    """Synfire chain detection via cross-correlation peak ordering. Abeles 1991.

    Returns list of chains (ordered neuron indices with sequential activation).
    """
    n = len(trains)
    if n < min_chain_length:
        return []
    peak_lags = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cc, lags = cross_correlation(trains[i], trains[j], max_lag_ms=max_delay_ms, dt=dt)
            if cc.size > 0:
                peak_idx = np.argmax(cc)
                peak_lags[i, j] = lags[peak_idx]
    chains = []
    visited = set()
    for start in range(n):
        if start in visited:
            continue
        chain = [start]
        current = start
        for _ in range(n):
            candidates = []
            for j in range(n):
                if j in chain:
                    continue
                if 0 < peak_lags[current, j] <= max_delay_ms:
                    candidates.append((peak_lags[current, j], j))
            if not candidates:
                break
            candidates.sort()
            nxt = candidates[0][1]
            chain.append(nxt)
            current = nxt
        if len(chain) >= min_chain_length:
            chains.append(chain)
            visited.update(chain)
    return chains
