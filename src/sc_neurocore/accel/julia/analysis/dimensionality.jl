# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis/dimensionality
#
# PCA, demixed PCA (Kobak et al. 2016) and factor analysis (Rubin & Thayer 1982)
# on pre-binned, mean-centred matrices. The covariance eigendecomposition uses
# the native LAPACK symmetric solver with descending eigenvalues and
# sign-canonicalised eigenvectors; factor analysis starts from a deterministic
# PCA initialisation and solves its SPD systems by Cholesky. Matches the NumPy,
# Rust, Go and Mojo backends to floating-point round-off.

module DimensionalityAccel

using Statistics, LinearAlgebra

"""Descending eigenvalues + sign-canonicalised eigenvectors of `Symmetric(cov)`."""
function _eig_desc(cov)
    e = eigen(Symmetric(cov))
    order = sortperm(e.values; rev = true)
    vals = e.values[order]
    vecs = Matrix(e.vectors[:, order])
    for c in 1:size(vecs, 2)
        piv = argmax(abs.(@view vecs[:, c]))
        if vecs[piv, c] < 0
            @views vecs[:, c] .= -vecs[:, c]
        end
    end
    return vals, vecs
end

"""PCA of a centred `(n_neurons, n_bins)` matrix → `(projected, explained)`."""
function pca_from_matrix(mat_in, n_components)
    mat = Matrix{Float64}(mat_in)
    d, t = size(mat)
    cov = (mat * mat') / max(t - 1, 1)
    vals, vecs = _eig_desc(cov)
    nc = min(Int(n_components), d)
    total = sum(vals)
    explained = total > 0 ? vals[1:nc] ./ total : vals[1:nc]
    projected = vecs[:, 1:nc]' * mat
    return projected, collect(explained)
end

"""Demixed PCA of a centred `(n_conditions, n_bins)` matrix."""
function demixed_from_matrix(mean_mat_in, n_components)
    mean_mat = Matrix{Float64}(mean_mat_in)
    n_cond, t = size(mean_mat)
    cov = (mean_mat' * mean_mat) / n_cond
    vals, vecs = _eig_desc(cov)
    nc = min(Int(n_components), t)
    total = sum(vals)
    explained = total > 0 ? vals[1:nc] ./ total : vals[1:nc]
    projected = mean_mat * vecs[:, 1:nc]
    return projected, collect(explained)
end

"""Factor analysis EM of a centred `(n_neurons, n_bins)` matrix (deterministic init)."""
function factor_analysis(mat_in, n_factors, n_iter)
    mat = Matrix{Float64}(mat_in)
    d, t = size(mat)
    cov = (mat * mat') / t
    vals, vecs = _eig_desc(cov)
    nf = min(Int(n_factors), d)
    loadings = vecs[:, 1:nf] .* sqrt.(max.(vals[1:nf], 0.0))'
    psi = diag(cov)
    eyenf = Matrix{Float64}(I, nf, nf)
    for _ in 1:Int(n_iter)
        psi_inv = 1.0 ./ (psi .+ 1e-10)
        m = loadings' * (psi_inv .* loadings) + eyenf
        cf = cholesky(Symmetric(m))
        m_inv = cf \ eyenf
        beta = cf \ Matrix((loadings .* psi_inv)')
        ez = beta * mat
        ezzt = nf * m_inv + (ez * ez') / t
        mat_ez_t = (mat * ez') / t
        cf2 = cholesky(Symmetric(ezzt))
        loadings = Matrix((cf2 \ Matrix(mat_ez_t'))')
        psi = max.(diag(cov - (loadings * ez) * mat' / t), 1e-6)
    end
    return loadings, psi
end

end # module DimensionalityAccel
