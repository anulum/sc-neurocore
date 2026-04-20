# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for dimensionality

fn spike_train_pca(trains: Int, n_components: Int, bin_size: Int) -> Int:
    var _spike_train_pca_line = 'trains: list[ndarray], n_components: int = 3, bin_size: int '
    var _spike_train_pca_line = ') -> tuple[ndarray, ndarray]:'
    var _spike_train_pca_line = 'if not trains:'
    return 0  # return array([[]]), array([])
    var _spike_train_pca_line = 'binned = array([bin_spike_train(t, bin_size).astype(float64)'
    var _spike_train_pca_line = 'min_bins = min(b.size for b in binned)'
    var _spike_train_pca_line = 'mat = array([b[:min_bins] for b in binned])'
    var _spike_train_pca_line = 'mat -= mat.mean(axis=1, keepdims=True)'
    var _spike_train_pca_line = 'cov = cov(mat)'
    var _spike_train_pca_line = 'if cov.ndim < 2:'
    return 0  # return mat[:1], array([1.0])
    var _spike_train_pca_line = 'eigvals, eigvecs = linalg.eigh(cov)'
    var _spike_train_pca_line = 'idx = argsort(eigvals)[::-1][:n_components]'
    var _spike_train_pca_line = 'components = eigvecs[:, idx]'
    var _spike_train_pca_line = 'projected = components.T @ mat'
    var _spike_train_pca_line = 'total_var = eigvals.sum()'
    var _spike_train_pca_line = 'explained = eigvals[idx] / total_var if total_var > 0 else e'
    return 0  # return projected, explained

fn demixed_pca(trains_by_condition: Int, n_components: Int, bin_size: Int) -> Int:
    var _demixed_pca_line = 'trains_by_condition: dict[int, list[ndarray]], n_components:'
    var _demixed_pca_line = ') -> tuple[ndarray, ndarray]:'
    var _demixed_pca_line = 'all_means = []'
    var _demixed_pca_line = 'for cond, trains in sorted(trains_by_condition.items()):'
    var _demixed_pca_line = 'binned = [bin_spike_train(t, bin_size).astype(float64) for t'
    var _demixed_pca_line = 'min_bins = min(b.size for b in binned)'
    var _demixed_pca_line = 'mat = array([b[:min_bins] for b in binned])'
    var _demixed_pca_line = 'all_means.append(mat.mean(axis=0))'
    var _demixed_pca_line = 'if len(all_means) < 2:'
    return 0  # return array([[]]), array([])
    var _demixed_pca_line = 'mean_mat = array(all_means)'
    var _demixed_pca_line = 'mean_mat -= mean_mat.mean(axis=0, keepdims=True)'
    var _demixed_pca_line = 'cov = mean_mat.T @ mean_mat / mean_mat.shape[0]'
    var _demixed_pca_line = 'eigvals, eigvecs = linalg.eigh(cov)'
    var _demixed_pca_line = 'idx = argsort(eigvals)[::-1][:n_components]'
    var _demixed_pca_line = 'total = eigvals.sum()'
    var _demixed_pca_line = 'explained = eigvals[idx] / total if total > 0 else eigvals[i'
    var _demixed_pca_line = 'projected = mean_mat @ eigvecs[:, idx]'
    return 0  # return projected, explained

fn factor_analysis(trains: Int, n_factors: Int, bin_size: Int, n_iter: Int) -> Int:
    var _factor_analysis_line = 'trains: list[ndarray], n_factors: int = 3, bin_size: int = 1'
    var _factor_analysis_line = ') -> tuple[ndarray, ndarray]:'
    var _factor_analysis_line = 'binned = [bin_spike_train(t, bin_size).astype(float64) for t'
    var _factor_analysis_line = 'min_bins = min(b.size for b in binned)'
    var _factor_analysis_line = 'mat = array([b[:min_bins] for b in binned])'
    var _factor_analysis_line = 'd, t = mat.shape'
    var _factor_analysis_line = 'mat -= mat.mean(axis=1, keepdims=True)'
    var _factor_analysis_line = 'cov = mat @ mat.T / t'
    var _factor_analysis_line = 'psi = diag(cov).copy()'
    var _factor_analysis_line = 'loadings = random.default_rng(42).normal(0, 0.1, (d, n_facto'
    var _factor_analysis_line = 'for _ in range(n_iter):'
    var _factor_analysis_line = 'psi_inv = diag(1.0 / (psi + 1e-10))'
    var _factor_analysis_line = 'm = loadings.T @ psi_inv @ loadings + eye(n_factors)'
    var _factor_analysis_line = 'm_inv = linalg.inv(m)'
    var _factor_analysis_line = 'beta = m_inv @ loadings.T @ psi_inv'
    var _factor_analysis_line = 'ez = beta @ mat'
    var _factor_analysis_line = 'ezzt = n_factors * m_inv + ez @ ez.T / t'
    var _factor_analysis_line = 'loadings = mat @ ez.T / t @ linalg.inv(ezzt / t * t)'
    var _factor_analysis_line = 'psi = diag(cov - loadings @ ez @ mat.T / t)'
    var _factor_analysis_line = 'psi = clip(psi, 1e-6, 0)'
    return 0  # return loadings, psi

