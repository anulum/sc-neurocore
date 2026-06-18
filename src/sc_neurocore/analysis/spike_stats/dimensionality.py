# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dimensionality reduction for spike train populations

"""Dimensionality reduction for spike train populations."""

from __future__ import annotations

from typing import Any
import numpy as np

from .basic import bin_spike_train


def spike_train_pca(
    trains: list[np.ndarray[Any, Any]], n_components: int = 3, bin_size: int = 10
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """PCA on binned spike count matrix (neurons x time_bins).

    Returns (projected, explained_variance_ratio).
    """
    if not trains:
        return np.array([[]]), np.array([])
    binned = np.array([bin_spike_train(t, bin_size).astype(np.float64) for t in trains])
    min_bins = min(b.size for b in binned)
    mat = np.array([b[:min_bins] for b in binned])
    mat -= mat.mean(axis=1, keepdims=True)
    cov = np.cov(mat)
    if cov.ndim < 2:
        return mat[:1], np.array([1.0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:n_components]
    components = eigvecs[:, idx]
    projected = components.T @ mat
    total_var = eigvals.sum()
    explained = eigvals[idx] / total_var if total_var > 0 else eigvals[idx]
    return projected, explained


def demixed_pca(
    trains_by_condition: dict[int, list[np.ndarray[Any, Any]]],
    n_components: int = 3,
    bin_size: int = 10,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Demixed PCA. Kobak et al. 2016.

    Separates condition-dependent and condition-independent variance.
    trains_by_condition: {condition_id: [list of binary trains per neuron]}.
    Returns (projected, explained_variance_ratio).
    """
    all_means = []
    for cond, trains in sorted(trains_by_condition.items()):
        binned = [bin_spike_train(t, bin_size).astype(np.float64) for t in trains]
        min_bins = min(b.size for b in binned)
        mat = np.array([b[:min_bins] for b in binned])
        all_means.append(mat.mean(axis=0))
    if len(all_means) < 2:
        return np.array([[]]), np.array([])
    mean_mat = np.array(all_means)
    mean_mat -= mean_mat.mean(axis=0, keepdims=True)
    cov = mean_mat.T @ mean_mat / mean_mat.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:n_components]
    total = eigvals.sum()
    explained = eigvals[idx] / total if total > 0 else eigvals[idx]
    projected = mean_mat @ eigvecs[:, idx]
    return projected, explained


def factor_analysis(
    trains: list[np.ndarray[Any, Any]], n_factors: int = 3, bin_size: int = 10, n_iter: int = 50
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Factor analysis via EM. Rubin & Thayer 1982.

    Returns (loading_matrix [n_neurons x n_factors], uniquenesses [n_neurons]).
    """
    binned = [bin_spike_train(t, bin_size).astype(np.float64) for t in trains]
    min_bins = min(b.size for b in binned)
    mat = np.array([b[:min_bins] for b in binned])
    d, t = mat.shape
    mat -= mat.mean(axis=1, keepdims=True)
    cov = mat @ mat.T / t
    psi = np.diag(cov).copy()
    loadings = np.random.default_rng(42).normal(0, 0.1, (d, n_factors))
    for _ in range(n_iter):
        psi_inv = np.diag(1.0 / (psi + 1e-10))
        m = loadings.T @ psi_inv @ loadings + np.eye(n_factors)
        m_inv = np.linalg.inv(m)
        beta = m_inv @ loadings.T @ psi_inv
        ez = beta @ mat
        ezzt = n_factors * m_inv + ez @ ez.T / t
        loadings = mat @ ez.T / t @ np.linalg.inv(ezzt / t * t)
        psi = np.diag(cov - loadings @ ez @ mat.T / t)
        psi = np.clip(psi, 1e-6, None)
    return loadings, psi
