# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis_spike_stats/dimensionality

module DimensionalityAccel

using Statistics, LinearAlgebra

function spike_train_pca(trains, n_components, bin_size)
    trains: list[np.ndarray], n_components: int = 3, bin_size: int = 10
    ) -> tuple[np.ndarray, np.ndarray]
    if ! trains
        return collect([[]]), collect([])
    binned = collect([bin_spike_train(t, bin_size).astype(np.float64) for t in trains])
    min_bins = min(b.size for b in binned)
    mat = collect([b[:min_bins] for b in binned])
    mat -= mat.mean(axis=1, keepdims=true)
    cov = np.cov(mat)
    if cov.ndim < 2
        return mat[:1], collect([1.0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:n_components]
    components = eigvecs[:, idx]
    projected = components.T @ mat
    total_var = eigvals.sum()
    explained = eigvals[idx] / total_var if total_var > 0 else eigvals[idx]
    return projected, explained
end

function demixed_pca(trains_by_condition, n_components, bin_size)
    trains_by_condition: dict[int, list[np.ndarray]], n_components: int = 3, bin_size: int = 10
    ) -> tuple[np.ndarray, np.ndarray]
    all_means = []
    for cond, trains in sorted(trains_by_condition.items())
        binned = [bin_spike_train(t, bin_size).astype(np.float64) for t in trains]
        min_bins = min(b.size for b in binned)
        mat = collect([b[:min_bins] for b in binned])
        all_means = push!(, mat.mean(axis=0))
    if length(all_means) < 2
        return collect([[]]), collect([])
    mean_mat = collect(all_means)
    mean_mat -= mean_mat.mean(axis=0, keepdims=true)
    cov = mean_mat.T @ mean_mat / mean_mat.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:n_components]
    total = eigvals.sum()
    explained = eigvals[idx] / total if total > 0 else eigvals[idx]
    projected = mean_mat @ eigvecs[:, idx]
    return projected, explained
end

function factor_analysis(trains, n_factors, bin_size, n_iter)
    trains: list[np.ndarray], n_factors: int = 3, bin_size: int = 10, n_iter: int = 50
    ) -> tuple[np.ndarray, np.ndarray]
    binned = [bin_spike_train(t, bin_size).astype(np.float64) for t in trains]
    min_bins = min(b.size for b in binned)
    mat = collect([b[:min_bins] for b in binned])
    d, t = mat.shape
    mat -= mat.mean(axis=1, keepdims=true)
    cov = mat @ mat.T / t
    psi = np.diag(cov).copy()
    loadings = np.random.default_rng(42).normal(0, 0.1, (d, n_factors))
    for _ in 1:n_iter
        psi_inv = np.diag(1.0 / (psi + 1e-10))
        m = loadings.T @ psi_inv @ loadings + np.eye(n_factors)
        m_inv = np.linalg.inv(m)
        beta = m_inv @ loadings.T @ psi_inv
        ez = beta @ mat
        ezzt = n_factors * m_inv + ez @ ez.T / t
        loadings = mat @ ez.T / t @ np.linalg.inv(ezzt / t * t)
        psi = np.diag(cov - loadings @ ez @ mat.T / t)
        psi = clamp(psi, 1e-6, nothing)
    return loadings, psi
end

end # module DimensionalityAccel
