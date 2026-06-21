# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis/sorting_quality
#
# Mahalanobis cluster-quality metrics (Harris et al. 2001; Schmitzer-Torbert et
# al. 2005). The squared Mahalanobis distance uses the native LAPACK Cholesky
# factor of the regularised cluster covariance (the covariance is never inverted
# explicitly), matching the NumPy, Rust, Go and Mojo backends within float64
# round-off.

module SortingQualityAccel

using Statistics, LinearAlgebra

"""
    _mahalanobis_sq(cluster, noise) -> Vector{Float64}

Squared Mahalanobis distances of each row of `noise` (n_noise × d) from the
`cluster` mean (n_cluster × d), via the Cholesky factor of the jitter-regularised
cluster covariance.
"""
function _mahalanobis_sq(cluster, noise)
    cl = Matrix{Float64}(cluster)
    ns = Matrix{Float64}(noise)
    d = size(cl, 2)
    mu = vec(mean(cl; dims = 1))
    covm = cov(cl; dims = 1) + 1e-8 * Matrix{Float64}(I, d, d)
    chol = cholesky(Symmetric(covm))
    n_noise = size(ns, 1)
    out = Vector{Float64}(undef, n_noise)
    @inbounds for i in 1:n_noise
        z = chol.L \ (vec(ns[i, :]) .- mu)
        out[i] = sum(abs2, z)
    end
    return out
end

"""Isolation distance (Harris et al. 2001)."""
function isolation_distance(cluster, noise)
    n_c = size(cluster, 1)
    if n_c < 2 || size(noise, 1) < n_c
        return NaN
    end
    mah = sort(_mahalanobis_sq(cluster, noise))
    return n_c <= length(mah) ? mah[n_c] : mah[end]
end

"""L-ratio cluster quality (Schmitzer-Torbert et al. 2005)."""
function l_ratio(cluster, noise)
    n_c = size(cluster, 1)
    if n_c < 2 || size(noise, 1) == 0
        return NaN
    end
    d = size(cluster, 2)
    s = 0.0
    for m in _mahalanobis_sq(cluster, noise)
        s += clamp(exp(-0.5 * (max(m, 1e-10) - d)), 0.0, 1.0)
    end
    return s / n_c
end

end # module SortingQualityAccel
